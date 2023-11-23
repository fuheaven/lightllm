import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
import triton
from functools import partial

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama_awquant2.layer_weights.transformer_layer_weight import LlamaTransformerLayerActivationWeightQuantized2
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import TransformerLayerInferTpl

from lightllm.common.basemodel.cuda_kernel.ppl_awquant import matmul_i8_i32_ppl, skiprmsnorm_ppl, channel_token_dequant_i32_fp16_ppl
from lightllm.common.basemodel.cuda_kernel.ppl_awquant import gatesilu_i32_i8_ppl
from lightllm.utils.infer_utils import mark_cost_time

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
        self._bind_norm()
        self._bind_matmul()   
        self._bind_silu()     
        self._bind_attention()
        return

    def _bind_norm(self):
        self._att_norm = partial(LlamaTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(LlamaTransformerLayerInfer._ffn_norm, self)
        if "ppl_int8_activation_weight" in self.mode:
            self._awquant_att_norm = partial(LlamaTransformerLayerInferAWquant2._awquant_att_norm_ppl_int8, self)
            self._awquant_ffn_norm = partial(LlamaTransformerLayerInferAWquant2._awquant_ffn_norm_ppl_int8, self)
        else:
            raise Exception(f"error mode {self.mode}")
        return
    
    def _bind_matmul(self):
        if "ppl_int8_activation_weight" in self.mode:
            self._awquant_matmul_for_qkv = partial(LlamaTransformerLayerInferAWquant2._awquant_matmul_ppl_int8_quant_dequant, self)
            self._awquant_matmul_for_o = partial(LlamaTransformerLayerInferAWquant2._awquant_matmul_ppl_int8_quant_dequant, self)
            self._awquant_matmul_for_ffn_up = partial(LlamaTransformerLayerInferAWquant2._awquant_matmul_ppl_int8_quant, self)
            self._awquant_matmul_for_ffn_down = partial(LlamaTransformerLayerInferAWquant2._awquant_matmul_ppl_int8_quant_dequant, self)
            if self.tp_rank_ == 0 and self.layer_num_ == 0:
                print("model use ppl_int8_activation_weight kernel")
        else:
            raise Exception(f"error mode {self.mode}")
        return

    def _bind_silu(self):
        if "ppl_int8_activation_weight" in self.mode:
            func = partial(LlamaTransformerLayerInferAWquant2._awquant_silu_ppl_int8, self)
            self._awquant_silu = func
        else:
            raise Exception(f"error mode {self.mode}")
        return
    
    def _bind_attention(self):
        self._context_attention_kernel = partial(LlamaTransformerLayerInfer._context_attention_kernel, self)
        if "ppl_int8kv" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_ppl_int8kv, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_ppl_int8kv, self)
        elif "triton_int8kv" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_int8kv, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_int8kv, self)
        elif "triton_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_flashdecoding, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)   
        else:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_normal, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        return

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
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        # o_tensor = self._awquant_matmul_for_o(input, 
        #                                      quant_weight_params=layer_weight.o_weight_,
        #                                      is_prefill=infer_state.is_prefill)
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

    def _awquant_matmul_ppl_int8_quant_dequant(self, input, quant_weight_params, is_prefill, token_scale=None, out=None, bias=None, has_act=False):
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