import onnx
from onnx import helper


def make_affine_grid():
    node = helper.make_node('AffineGrid',
                            inputs=['x', 'w'],
                            outputs=['y'],
                            name='test_affine_grid_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_affine_grid_1",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 3]),
                helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 4])],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 8
    onnx.save(model, "./test_affine_grid_case_1.onnx")
    onnx.checker.check_model(model)

def make_affine_grid1():
    node = helper.make_node('AffineGrid',
                            inputs=['x', 'w'],
                            outputs=['y'],
                            align_corners=1,
                            name='test_affine_grid_2')
    graph = helper.make_graph(
        nodes=[node],
        name="test_affine_grid_2",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 3]),
                helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 4])],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_affine_grid_case_2.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_affine_grid1()
    make_affine_grid()
    
