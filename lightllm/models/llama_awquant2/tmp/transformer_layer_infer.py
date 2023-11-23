import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
import triton
from functools import partial

from lightllm.models.llama_awquant2.layer_weights.transformer_layer_weight import LlamaTransformerLayerActivationWeightQuantized2
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.common.basemodel import TransformerLayerInferTpl

from lightllm.common.basemodel.cuda_kernel.ppl_awquant import matmul_i8_i32_ppl, skiprmsnorm_ppl, channel_token_dequant_i32_fp16_ppl
from lightllm.common.basemodel.cuda_kernel.ppl_awquant import dynamic_channelwise_quant_fp16_i8_ppl, gatesilu_i32_i8_ppl
from lightllm.utils.infer_utils import mark_cost_time

def cosine(a, b):
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
    a = a.flatten().astype('float32')
    b = b.flatten().astype('float32')
    u = np.sum(a * b)
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return u / d

class LlamaTransformerLayerInferAWquant2(TransformerLayerInferTpl):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self.inter_dim_ = network_config['intermediate_size']
        self._bind_func()
        return
    
    def _bind_func(self):
        self._awquant_matmul_for_qkv = self._awquant_matmul_ppl_int8_quant_dequant
        self._awquant_matmul_for_o = self._awquant_matmul_ppl_int8_quant_dequant
        self._awquant_matmul_for_ffn_up = self._awquant_matmul_ppl_int8_quant
        self._awquant_matmul_for_ffn_down = self._awquant_matmul_ppl_int8_quant_dequant
        self._awquant_silu = self._awquant_silu_ppl_int8
        self._awquant_att_norm = self._awquant_att_norm_ppl_int8
        self._awquant_ffn_norm = self._awquant_ffn_norm_ppl_int8
        if "ppl_int8kv" in self.mode:
            self._token_attention_kernel = self._token_decode_attention_ppl_int8kv
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_ppl_int8kv
        elif "triton_int8kv" in self.mode:
            self._token_attention_kernel = self._token_decode_attention_int8kv
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_int8kv
        elif "triton_flashdecoding" in self.mode:
            self._token_attention_kernel = self._token_decode_attention_flashdecoding
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_normal   
        else:
            self._token_attention_kernel = self._token_decode_attention_normal
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_normal   
        if self.tp_rank_ == 0 and self.layer_num_ == 0:
            print("model use ppl_int8_activation_weight kernel")
        return
    
    def _att_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerActivationWeightQuantized2)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_)
    
    def _ffn_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerActivationWeightQuantized2)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_)

    def _get_qkv_origin(self, input, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerActivationWeightQuantized2)->torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.k_weight_,
                    out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.v_weight_,
                    out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q

    def _get_qkv_int8_fused(self, input, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerActivationWeightQuantized2, token_scale):
        qkv_output = self._awquant_matmul_for_qkv(input.view(-1, self.embed_dim_), 
                                                    quant_weight_params=layer_weight.qkv_weight_,
                                                    is_prefill=infer_state.is_prefill,
                                                    token_scale=token_scale)
        
        tp_k_head_dim = self.tp_k_head_num_ * self.head_dim_
        q = qkv_output[:, : -2 * tp_k_head_dim]
        k = qkv_output[:, -2 * tp_k_head_dim: -tp_k_head_dim]
        v = qkv_output[:, -tp_k_head_dim :]

        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

    def _get_qkv_int8(self, input, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerActivationWeightQuantized2, token_scale):
        q = self._awquant_matmul_for_qkv(input.view(-1, self.embed_dim_),
                                            quant_weight_params=layer_weight.q_weight_int8,
                                            is_prefill=infer_state.is_prefill,
                                            token_scale=token_scale)
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)

        out = self._awquant_matmul_for_qkv(input.view(-1, self.embed_dim_),
                                            quant_weight_params=layer_weight.k_weight_int8,
                                            is_prefill=infer_state.is_prefill,
                                            token_scale=token_scale)
        cache_k_ = out.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        out = self._awquant_matmul_for_qkv(input.view(-1, self.embed_dim_),
                                            quant_weight_params=layer_weight.v_weight_int8,
                                            is_prefill=infer_state.is_prefill,
                                            token_scale=token_scale)
        cache_v_ = out.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_


    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1, token_scale, skip_out = self._awquant_att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_int8_fused(input1, cache_k, cache_v, infer_state, layer_weight, token_scale)

        # input1 = self._att_norm(input_embding, infer_state, layer_weight)
        # cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        # q = self._get_qkv_origin(input1, cache_k, cache_v, infer_state, layer_weight)

        # print(cosine(q, q_origin))
        # print(cosine(cache_k, cache_k_origin))
        # print(cosine(cache_v, cache_v_origin))
        # import pdb ; pdb.set_trace()

        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o_kernel = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o(o_kernel, infer_state, layer_weight)

        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1, token_scale, skip_out = self._awquant_att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_int8_fused(input1, cache_k, cache_v, infer_state, layer_weight, token_scale)

        # input1 = self._att_norm(input_embding, infer_state, layer_weight)
        # cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        # q = self._get_qkv_origin(input1, cache_k, cache_v, infer_state, layer_weight)

        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o_kernel = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o_kernel, infer_state, layer_weight)

        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return
    
    def _context_attention_kernel(self, q, k, v, infer_state:LlamaInferStateInfo, layer_weight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor

    def _get_o(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerActivationWeightQuantized2)->torch.Tensor:
        # o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        o_tensor = self._awquant_matmul_for_o(input, 
                                             quant_weight_params=layer_weight.o_weight_,
                                             is_prefill=infer_state.is_prefill)
        return o_tensor

    def _get_o_origin(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerActivationWeightQuantized2)->torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_origin)
        return o_tensor

    def _ffn(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerActivationWeightQuantized2)->torch.Tensor:
        gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_proj_origin)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.up_proj_origin)
        input = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj_origin)
        ffn1_out = None
        return ffn2_out

    def _ffn_int8_fused(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerActivationWeightQuantized2, token_scale)->torch.Tensor:
        gate_up_output = self._awquant_matmul_for_ffn_up(input.view(-1, self.embed_dim_),
                                                        layer_weight.gate_up_proj,
                                                        is_prefill=infer_state.is_prefill)
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        _, gate_up_scale = layer_weight.gate_up_proj
        gate_up_scale = gate_up_scale.view(2, tp_inter_dim)
        ffn1_out, ffn1_out_scale = self._awquant_silu(gate_up_output[:, 0], gate_up_output[:, 1], 
                                        gate_up_scale[0], gate_up_scale[1], token_scale)
        gate_up_output = None
        ffn2_out = self._awquant_matmul_for_ffn_down(ffn1_out, 
                                                    layer_weight.down_proj,
                                                    is_prefill=infer_state.is_prefill,
                                                    token_scale=ffn1_out_scale)
        ffn1_out = None
        return ffn2_out

    def _ffn_int8(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerActivationWeightQuantized2, token_scale) -> torch.Tensor:
        gate_out = self._awquant_matmul_for_ffn_up(input.view(-1, self.embed_dim_), 
                                                layer_weight.gate_proj,
                                                is_prefill=infer_state.is_prefill,)
        up_out = self._awquant_matmul_for_ffn_up(input.view(-1, self.embed_dim_), 
                                                layer_weight.up_proj,
                                                is_prefill=infer_state.is_prefill,)
        input = None
        _, gate_proj_scale = layer_weight.gate_proj
        _, up_proj_scale = layer_weight.up_proj
        ffn1_out, ffn1_out_scale = self._awquant_silu(gate_out, up_out, 
                                        gate_proj_scale, up_proj_scale, token_scale)
        gate_out, up_out = None, None
        ffn2_out = self._awquant_matmul_for_ffn_down(ffn1_out, layer_weight.down_proj,
                                                    is_prefill=infer_state.is_prefill, 
                                                    token_scale=ffn1_out_scale)
        ffn1_out = None

        return ffn2_out
    
    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1, token_scale, skip_out = self._awquant_ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_int8_fused(input1, infer_state, layer_weight, token_scale)

        # input1_origin = self._ffn_norm(input_embdings, infer_state, layer_weight)
        # ffn_out_origin = self._ffn(input1_origin, infer_state, layer_weight)

        # print(cosine(ffn_out, ffn_out_origin))
        # import pdb ; pdb.set_trace()

        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1, token_scale, skip_out = self._awquant_ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_int8_fused(input1, infer_state, layer_weight, token_scale)

        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    def _copy_kv_to_mem_cache_normal(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
        return
    
    def _copy_kv_to_mem_cache_int8kv(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_quantize_kv(key_buffer,
                                    mem_index,
                                    mem_manager.key_buffer[self.layer_num_],
                                    mem_manager.key_scale_buffer[self.layer_num_])
        destindex_copy_quantize_kv(value_buffer,
                                    mem_index,
                                    mem_manager.value_buffer[self.layer_num_],
                                    mem_manager.value_scale_buffer[self.layer_num_])
        return
    
    def _copy_kv_to_mem_cache_ppl_int8kv(self, key_buffer, value_buffer, mem_index, mem_manager):
        from lightllm.models.llama.triton_kernel.ppl_quant_copy_kv import destindex_copy_quantize_kv
        destindex_copy_quantize_kv(key_buffer,
                                    mem_index,
                                    mem_manager.key_buffer[self.layer_num_],
                                    mem_manager.key_scale_buffer[self.layer_num_])
        destindex_copy_quantize_kv(value_buffer,
                                    mem_index,
                                    mem_manager.value_buffer[self.layer_num_],
                                    mem_manager.value_scale_buffer[self.layer_num_])
        return
    
    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.req_manager.req_to_token_indexs,
                      infer_state.b_req_idx,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)
        
        if triton.__version__ == "2.0.0":
            prob = torch.empty_like(att_m_tensor)
            token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
            att_m_tensor = None

            o_tensor = torch.empty_like(q)

            token_att_fwd2(prob,
                        infer_state.mem_manager.value_buffer[self.layer_num_],
                        o_tensor.view(calcu_shape1),
                        infer_state.req_manager.req_to_token_indexs,
                        infer_state.b_req_idx,
                        infer_state.b_start_loc,
                        infer_state.b_seq_len)
            prob = None
            return o_tensor
        elif triton.__version__ >= "2.1.0":
            o_tensor = torch.empty_like(q)
            from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
            token_softmax_reducev_fwd(att_m_tensor, 
                                      infer_state.mem_manager.value_buffer[self.layer_num_],
                                      o_tensor.view(calcu_shape1),
                                      infer_state.req_manager.req_to_token_indexs,
                                      infer_state.b_req_idx,
                                      infer_state.b_start_loc,
                                      infer_state.b_seq_len,
                                      infer_state.other_kv_index)
            return o_tensor
        else:
            raise Exception("not support triton version")

    def _token_decode_attention_int8kv(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")
        token_att_fwd_int8k(q.view(calcu_shape1),
                            infer_state.mem_manager.key_buffer[self.layer_num_],
                            infer_state.mem_manager.key_scale_buffer[self.layer_num_],
                            att_m_tensor,
                            infer_state.req_manager.req_to_token_indexs,
                            infer_state.b_req_idx,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)

        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)
        token_att_fwd2_int8v(prob,
                                infer_state.mem_manager.value_buffer[self.layer_num_],
                                infer_state.mem_manager.value_scale_buffer[self.layer_num_],
                                o_tensor.view(calcu_shape1),
                                infer_state.req_manager.req_to_token_indexs,
                                infer_state.b_req_idx,
                                infer_state.b_start_loc,
                                infer_state.b_seq_len,
                                infer_state.max_len_in_batch)
        prob = None
        return o_tensor
    
    def _token_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        from lightllm.models.llama.triton_kernel.flash_decoding import token_decode_attention_flash_decoding
        cache_k = infer_state.mem_manager.key_buffer[self.layer_num_]
        cache_v = infer_state.mem_manager.value_buffer[self.layer_num_]
        return token_decode_attention_flash_decoding(q, infer_state, self.tp_q_head_num_, self.head_dim_, cache_k, cache_v)
    
    def _token_decode_attention_ppl_int8kv(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        o_tensor = torch.empty_like(q)

        from lightllm_ppl_kernel import group8_int8kv_decode_attention
        import pdb ; pdb.set_trace()
        # group_int8kv_decode_attention(at::Tensor o, at::Tensor q, at::Tensor k, at::Tensor k_s,  at::Tensor v,  at::Tensor v_s, at::Tensor b_loc, at::Tensor b_seq_len, int max_len_in_batch)
        group8_int8kv_decode_attention(o_tensor.view(calcu_shape1),
                                                          q.view(calcu_shape1),
                                                          infer_state.mem_manager.key_buffer[self.layer_num_],
                                                          infer_state.mem_manager.key_scale_buffer[self.layer_num_],
                                                          infer_state.mem_manager.value_buffer[self.layer_num_],
                                                          infer_state.mem_manager.value_scale_buffer[self.layer_num_],
                                                          infer_state.req_manager.req_to_token_indexs.to(torch.int8),
                                                          infer_state.b_req_idx.to(torch.int8),
                                                          infer_state.b_seq_len.to(torch.int8),
                                                          infer_state.max_len_in_batch)
           
        return o_tensor

    def _awquant_matmul_ppl_int8_quant_dequant(self, input, quant_weight_params, is_prefill, token_scale=None, out=None, bias=None, has_act=False):
        if input.dtype == torch.float16:
            input, token_scale = dynamic_channelwise_quant_fp16_i8_ppl(input.transpose(0, 1))
        assert has_act == False
        if is_prefill:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        else:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        out = channel_token_dequant_i32_fp16_ppl(out, token_scale, qscale)
        if bias is None:
            return out
        else:
            out.add_(bias)
            return out

    def _awquant_matmul_ppl_int8_quant(self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False):
        assert has_act == False
        if is_prefill:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        else:
            qweight, qscale = quant_weight_params
            out = matmul_i8_i32_ppl(input, qweight)
        if bias is None:
            return out
        else:
            out.add_(bias)
            return out

    def _awquant_att_norm_ppl_int8(self, input, infer_state:LlamaInferStateInfo, layer_weight):
        return skiprmsnorm_ppl(input, layer_weight.att_norm_weight_)

    def _awquant_ffn_norm_ppl_int8(self, input, infer_state:LlamaInferStateInfo, layer_weight):
        return skiprmsnorm_ppl(input, layer_weight.ffn_norm_weight_)

    def _awquant_silu_ppl_int8(self, x, y, x_scale, y_scale, token_scale):
        return gatesilu_i32_i8_ppl(x, y, x_scale, y_scale, token_scale)