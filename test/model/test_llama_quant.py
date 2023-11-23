import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestLlamaInfer(unittest.TestCase):

    def test_llama_infer(self):
        from lightllm.models.llama_wquant.model import LlamaTpPartModelWQuant
        test_model_inference(world_size=8, 
                             model_dir="/home/fuhaiwen1/models/llama-7b", 
                             model_class=LlamaTpPartModelWQuant, 
                             batch_size=1, 
                             input_len=1024, 
                             output_len=1024,
                             mode=["ppl_int4weight"])
        return


if __name__ == '__main__':
    unittest.main()