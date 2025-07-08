import os
import sys

import onnx
from onnx.utils import extract_model
from rename_onnx_model import rename_onnx_node_name
from convert import merge_model

def get_max_box_num(onnx_path, output_name):
    model = onnx.load(onnx_path)
    graph = model.graph

    for i, output in enumerate(graph.output):
        shape = output.type.tensor_type.shape
        if output.name == output_name:
            return shape.dim[1].dim_value, shape.dim[2].dim_value
            
    return None

if __name__ == '__main__':
    if 2 != len(sys.argv) and 4 != len(sys.argv):
        print("Usage:\n\tpython %s input_path [input_name output_name]"%(sys.argv[0]))
        sys.exit(-1)
    input_path = sys.argv[1]
    output_path = input_path.replace(".onnx", '_bm1684x.onnx')
    if 4 == len(sys.argv):
        input_names = [sys.argv[2]]
        output_names = [sys.argv[3]]
    else:
        input_names = ["images"]
        output_names = ["/detect/Mul_6_output_0"] #, "/detect/Transpose_output_0"]
        model_size = os.path.getsize(input_path) / (1024 * 1024)
        print("%s size : %.2f"%(input_path, model_size))
        if model_size < 18 and model_size > 17:
            output_names.append("/detect/Transpose_output_0") # n
        elif model_size < 45 and model_size > 44:
            output_names.append("/detect/Transpose_6_output_0") # s
        elif model_size < 100 and model_size > 99:
            output_names.append("/detect/Transpose_6_output_0") # m
        else:
            raise Exception("Not supported %s"%(input_path))
        
    extract_model(input_path, output_path, input_names, output_names)
    box_num, class_num = get_max_box_num(output_path, output_names[-1])
    print("box_num : %d"%(box_num))
    merge_model(output_path, output_names[-1], box_num, class_num)
    rename_onnx_node_name(output_path, [output_names[0]], ["pred_bboxes"], output_path)