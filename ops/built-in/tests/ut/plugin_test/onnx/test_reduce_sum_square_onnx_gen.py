import onnx
from onnx import helper

def make_reduce_sum_square_v1():
    node = helper.make_node('ReduceSumSquare',
                            inputs=['X'],
                            outputs=['Y'],
                            axes=[1],
                            name='test_reduce_sum_square_v1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_reduce_sum_square_v1",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [12, 4, 3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [12, 4, 3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 1
    onnx.save(model, "./test_reduce_sum_square_v1.onnx")
    onnx.checker.check_model(model)

def make_reduce_sum_square_v11():
    node = helper.make_node('ReduceSumSquare',
                            inputs=['X'],
                            outputs=['Y'],
                            axes=[0, 1],
                            name='test_reduce_sum_square_v11')
    graph = helper.make_graph(
        nodes=[node],
        name="test_reduce_sum_square_v11",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [224, 4, 3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [224, 4, 3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_reduce_sum_square_v11.onnx")
    onnx.checker.check_model(model)

def make_reduce_sum_square_v13():
    node = helper.make_node('ReduceSumSquare',
                            inputs=['X'],
                            outputs=['Y'],
                            axes=[2],                                  
                            name='test_reduce_sum_square_v13')
    graph = helper.make_graph(
        nodes=[node],
        name="test_reduce_sum_square_v13",
        inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [12, 4, 3])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [12, 4, 3])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_reduce_sum_square_v13.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_reduce_sum_square_v1()
    make_reduce_sum_square_v11()
    make_reduce_sum_square_v13()
