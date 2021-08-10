import onnx
from onnx import helper
import numpy as np

def make_upsample(mode):
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    scales = helper.make_tensor("scales", onnx.TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 3.0])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1, 4, 6])

    node = helper.make_node('Upsample',
                            inputs=['X', 'scales'],
                            outputs=['Y'],
                            mode=mode)
    graph = helper.make_graph([node],
                              "test_upsample_1",
                              [x],
                              [y],
                              [scales],)

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 9
    onnx.save(model_def, "test_upsample_" + mode + "_case_v11.onnx")

if __name__ == '__main__':
    for mode in ["nearest", "linear"]:
        make_upsample(mode)


