import onnx
from onnx import helper


def make_qe_quant():
    node = helper.make_node('AscendDequant',
                            inputs=['A', 'B'],
                            outputs=['C'],
                            relu_flag=1,
                            dtype=1,
                            sqrt_mode=1,
                            name='test_qe_quant_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_qe_quant_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [2, 3, 4]),
                helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT16, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT16, [2, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_qe_quant_1.onnx")
    onnx.checker.check_model(model)
  
def make_qe_quant_s16():
    node = helper.make_node('AscendDequantS16',
                            inputs=['A', 'B'],
                            outputs=['C'],
                            relu_flag=1,
                            name='test_qe_quant_s16_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_qe_quant_s16_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [2, 3, 4]),
                helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT16, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT16, [2, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_qe_quant_s16_1.onnx")
    onnx.checker.check_model(model)

def make_re_quant():
    node = helper.make_node('AscendRequant',
                            inputs=['A', 'B'],
                            outputs=['C'],
                            relu_flag=1,
                            name='test_re_quant_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_re_quant_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [2, 3, 4]),
                helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT16, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT16, [2, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_re_quant_1.onnx")
    onnx.checker.check_model(model)

def make_re_quant_s16():
    node = helper.make_node('AscendRequantS16',
                            inputs=['A', 'B'],
                            outputs=['C'],
                            relu_flag=1,
                            dual_output=1,
                            name='test_re_quant_s16_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_re_quant_s16_1",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [2, 3, 4]),
                helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT16, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT16, [2, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_re_quant_s16_1.onnx")
    onnx.checker.check_model(model)

if __name__ == '__main__':
    make_re_quant_s16()
    make_re_quant()
    make_qe_quant_s16()
    make_qe_quant()
