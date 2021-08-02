import onnx
from onnx import helper


def make_Lp_pool(index):
    if index == 1:
      node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[2, 2],
                            name='test_Lp_pool_1')
    elif index == 2:
      node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[5, 5],
                            strides=[3, 3],
                            auto_pad='NOTSET',
                            name='test_Lp_pool_2')
    elif index == 3:
      node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[3, 3],
                            pads=[2, 2, 2, 2],
                            name='test_Lp_pool_3')

    graph = helper.make_graph(
        nodes=[node],
        name="test_Lp_pool_1",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 31, 31])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_Lp_pool_case_{0}.onnx".format(index))

def make_Lp_pool_4():
    node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[2, 2, 2],
                            name='test_Lp_pool_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_Lp_pool_4",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 32, 32, 32])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 31, 31, 31])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_4")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_Lp_pool_case_4.onnx")


def make_Lp_pool_v(version):
    if version == 12:
      node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[3, 3],
                            pads=[2, 2, 2, 2],
                            name='test_Lp_pool_v12')
    elif version == 13:
      node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[3, 3],
                            pads=[2, 2, 2, 2],
                            name='test_Lp_pool_v13')
    elif version == 9:
      node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[5, 5],
                            strides=[3, 3],
                            auto_pad='NOTSET',
                            name='test_Lp_pool_v9')

    graph = helper.make_graph(
        nodes=[node],
        name="test_Lp_pool_v12",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 28, 28])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 30, 30])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = version
    onnx.save(model, "./test_Lp_pool_case_v{0}.onnx".format(version))

def make_Lp_pool_input_1d():
    node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[2, ],
                            strides=[1, ],
                            auto_pad='NOTSET',
                            count_include_pad=0,
                            pads=[0, 0],
                            name='test_Lp_pool_input_1d')
    graph = helper.make_graph(
        nodes=[node],
        name="test_Lp_pool_input_1d",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT, [1, 3, 32])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, [1, 3, 31])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = 9
    onnx.save(model, "./test_Lp_pool_case_input_1d.onnx")


def make_Lp_pool_aicpu():
    node = helper.make_node('LpPool',
                            inputs=['x'],
                            outputs=['y'],
                            kernel_shape=[65, 65],
                            pads=[2, 2, 2, 2],
                            name='test_Lp_pool_aicpu')
    graph = helper.make_graph(
        nodes=[node],
        name="test_Lp_pool_aicpu",
        inputs=[helper.make_tensor_value_info(
            "x", onnx.TensorProto.FLOAT16, [1, 3, 128, 128])],
        outputs=[helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT16, [1, 3, 68, 68])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_3")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_Lp_pool_case_aicpu.onnx")


if __name__ == '__main__':
    make_Lp_pool(1)
    make_Lp_pool(2)
    make_Lp_pool(3)
    make_Lp_pool_4()
    make_Lp_pool_v(9)
    make_Lp_pool_v(12)
    make_Lp_pool_v(13)
    make_Lp_pool_input_1d()
    make_Lp_pool_aicpu()
