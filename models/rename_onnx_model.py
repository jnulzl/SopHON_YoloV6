import argparse
import sys
import onnx

def rename_onnx_node_name(onnx_model_path, origin_names, new_names, save_file):
    model = onnx.load(onnx_model_path)
    output_tensor_names = set()
    for ipt in model.graph.input:
        output_tensor_names.add(ipt.name)
    for node in model.graph.node:
        for out in node.output:
            output_tensor_names.add(out)

    for origin_name in origin_names:
        if origin_name not in output_tensor_names:
            print("[ERROR] Cannot find tensor name '{}' in onnx model graph.".format(origin_name))
            sys.exit(-1)
    if len(set(origin_names)) < len(origin_names):
        print("[ERROR] There's dumplicate name in --origin_names, which is not allowed.")
        sys.exit(-1)
    if len(new_names) != len(origin_names):
        print("[ERROR] Number of --new_names must be same with the number of --origin_names.")
        sys.exit(-1)
    if len(set(new_names)) < len(new_names):
        print("[ERROR] There's dumplicate name in --new_names, which is not allowed.")
        sys.exit(-1)
    for new_name in new_names:
        if new_name in output_tensor_names:
            print("[ERROR] The defined new_name '{}' is already exist in the onnx model, which is not allowed.")
            sys.exit(-1)

    for i, ipt in enumerate(model.graph.input):
        if ipt.name in origin_names:
            idx = origin_names.index(ipt.name)
            model.graph.input[i].name = new_names[idx]

    for i, node in enumerate(model.graph.node):
        for j, ipt in enumerate(node.input):
            if ipt in origin_names:
                idx = origin_names.index(ipt)
                model.graph.node[i].input[j] = new_names[idx]
        for j, out in enumerate(node.output):
            if out in origin_names:
                idx = origin_names.index(out)
                model.graph.node[i].output[j] = new_names[idx]

    for i, out in enumerate(model.graph.output):
        if out.name in origin_names:
            idx = origin_names.index(out.name)
            model.graph.output[i].name = new_names[idx]
    
    onnx.checker.check_model(model)
    onnx.save(model, save_file)
    print("[Finished] The new model saved in {}".format(save_file))
    print("[DEBUG INFO] The inputs of new model: {}".format([x.name for x in model.graph.input]))
    print("[DEBUG INFO] The outputs of new model: {}".format([x.name for x in model.graph.output]))
    