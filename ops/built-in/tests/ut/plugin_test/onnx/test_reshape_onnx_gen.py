import onnx
from onnx import helper

def reshape():
    node = helper.make_node('Reshape',
        inputs=['data', 'shape'],
        outputs=['reshaped'],
        allowzero=0,
        name='test_reshaped_1')
    shape = helper.make_tensor(
            "shape", onnx.TensorProto.INT64, [3], [4, 2, 3])
    
    graph = helper.make_graph(
        [node],
        "test_reshaped_1",
        [helper.make_tensor_value_info(
            "data", onnx.TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info(
            "reshaped", onnx.TensorProto.FLOAT, [4, 2, 3])],
        [shape]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_reshape_case_1.onnx")

if __name__ == '__main__':
    reshape()
