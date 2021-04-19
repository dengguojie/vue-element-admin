import onnx
from onnx import helper


def make_average_pool_1():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[2, 2],
                            name='test_average_pool_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_1",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 31, 31])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_average_pool_case_1.onnx")


def make_average_pool_2():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[5, 5],
                            strides=[3, 3],
                            ceil_mode=0,
                            auto_pad='NOTSET',
                            count_include_pad=0,
                            name='test_average_pool_2')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_2",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 10, 10])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_2")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_average_pool_case_2.onnx")


def make_average_pool_3():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[3, 3],
                            pads=[2, 2, 2, 2],
                            name='test_average_pool_3')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_3",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 28, 28])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 30, 30])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_average_pool_case_3.onnx")

def make_average_pool_4():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[2, 2, 2],
                            name='test_average_pool_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_4",
        inputs=[helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 32, 32, 32])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 31, 31, 31])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_4")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_average_pool_case_4.onnx")


def make_average_pool_v12():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[3, 3],
                            pads=[2, 2, 2, 2],
                            name='test_average_pool_v12')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_v12",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 28, 28])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 30, 30])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = 12
    onnx.save(model, "./test_average_pool_case_v12.onnx")


def make_average_pool_v13():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[3, 3],
                            pads=[2, 2, 2, 2],
                            name='test_average_pool_v13')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_v13",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 28, 28])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 30, 30])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = 13
    onnx.save(model, "./test_average_pool_case_v13.onnx")
    
    
def make_average_pool_v9():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[5, 5],
                            strides=[3, 3],
                            auto_pad='NOTSET',
                            count_include_pad=0,
                            name='test_average_pool_v9')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_v9",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 10, 10])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_average_pool_case_v9.onnx")


def make_average_pool_input_1d():
    node = helper.make_node('AveragePool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[2, ],
                            strides=[1, ],
                            auto_pad='NOTSET',
                            count_include_pad=0,
                            pads=[0, 0],
                            name='test_average_pool_input_1d')
    graph = helper.make_graph(
        nodes=[node],
        name="test_average_pool_input_1d",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 32])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 31])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_average_pool_case_input_1d.onnx")


if __name__ == '__main__':
    make_average_pool_1()
    make_average_pool_2()
    make_average_pool_3()
    make_average_pool_4()
    make_average_pool_v9()
    make_average_pool_v12()
    make_average_pool_v13()
    make_average_pool_input_1d()
