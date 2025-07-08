import os
import sys
import onnx
import torch.onnx
import torch.nn as nn
import onnxsim

class PostProcess(nn.Module):
    def __init__(self, top_k):
        super(PostProcess, self).__init__()
        self.top_k = top_k
        pass

    def forward(self, cls_score_list):
        max_scores, max_index = cls_score_list.max(dim=-1, keepdim=False)
        sorted_max_scores, sorted_index = max_scores.topk(self.top_k, dim=-1)
        return sorted_max_scores, sorted_index.to(torch.float32), max_index.to(torch.float32)

def convert_output_float32(box_num, class_num, top_k=100):
    torch_model = PostProcess(top_k)
    x = torch.randn(1, box_num, class_num)
    output = torch_model(x)
    # print(output.shape, output.dtype)
    save_path = "post_process.onnx"
    dynamic_axes = {
        "transpose_input" :{
            0:'unk__0',
        },

        "topk_scores" :{
            0:'unk__0',
        },
        "topk_index" :{
            0:'unk__0',
        },
        "max_index" :{
            0:'unk__0',
        }
    }
        
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      (x),                         # model input (or a tuple for multiple inputs)
                      save_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ["transpose_input"],   # the model's input names
                      output_names = ["topk_scores","topk_index","max_index"], # the model's output names
                      # output_names = ['locs', 'max_val_x', 'max_val_y'] # the model's output names
                      dynamic_axes=dynamic_axes)

    #simplify ONNX...                          
    onnx_model = onnx.load(save_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, save_path)
    

def merge_model(onnx_path_1, output_name, box_num, class_num):
    
    onnx_model_1 = onnx.load(onnx_path_1)
    convert_output_float32(box_num, class_num)
    onnx_model_2 = onnx.load("post_process.onnx")
    io_map = [(output_name, "transpose_input")]

    onnx_merge = onnx.compose.merge_models(onnx_model_1, onnx_model_2, io_map, 
                                        doc_string="output_to_float32",
                                        producer_name="jnulzl",
                                        # prefix1="rtm_", 
                                        # prefix2="head_%s"%(io_map[0][1]), 
                                        )

    onnx.checker.check_model(onnx_merge)  # check onnx model
    onnx_merge, check = onnxsim.simplify(onnx_merge)
    onnx.save(onnx_merge, onnx_path_1)
    # onnx.save(onnx_merge, onnx_path_1.replace('.onnx', '_merge.onnx'))
    # onnx_path_1 = onnx_path_1.replace('.onnx', '_merge.onnx')
    
    
if __name__ == '__main__':
    
    if 3 != len(sys.argv):
        print("Usage:\n\tpython %s box_num class_num"%(sys.argv[0]))
        sys.exit(-1)
    box_num, class_num = int(sys.argv[1]), int(sys.argv[2])
    convert_output_float32(box_num, class_num)