import os
import json
import torch
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.mem_utils import select_mem_manager_class

from lightllm.models.llama_awquant2.layer_weights.transformer_layer_weight import LlamaTransformerLayerActivationWeightQuantized2
from lightllm.models.llama_awquant2.layer_infer.transformer_layer_infer import LlamaTransformerLayerInferAWquant2

class LlamaTpPartModelAWQuant2(LlamaTpPartModel):
    # weight class
    transformer_weight_class = LlamaTransformerLayerActivationWeightQuantized2

    # infer class
    transformer_layer_infer_class = LlamaTransformerLayerInferAWquant2

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _verify_params(self):
        assert any("int8_activation_weight" in mode_ or "int4_activation_weight" in mode_ for mode_ in self.mode), "only for weight quant model"
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(self.max_total_token_num, 
                                                     dtype=torch.float16,
                                                     head_num=self.config["num_key_value_heads"] // self.world_size_,
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"])
        return