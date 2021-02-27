#argmax.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_argmax(version_num):
    node = helper.make_node('ArgMax',
                            inputs=['data'],
                            outputs=['result'],
                            keepdims=1,
                            name='test_argmax_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_argmaxx_1",
        inputs=[helper.make_tensor_value_info(
            "data", onnx.TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info(
            "result", onnx.TensorProto.FLOAT, [1, 3, 4])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_argmax_case_V{}.onnx".format(version_num))

def export_keepdims_select_last_index(version_num):
    data = helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 2,])
    result = helper.make_tensor_value_info("result", onnx.TensorProto.FLOAT, [1, 1])

    axis = 1
    keepdims = 1

    node = helper.make_node(
        'ArgMax',
        inputs=['data'],
        outputs=['result'],
        axis=1,
        keepdims=1,
        select_last_index=True
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [data],
        [result],
    )

    model_def = onnx.helper.make_model(graph_def, producer_name='tr')
    model_def.opset_import[0].version = version_num
    onnx.save(model_def, "./test_argmax_case_V{}.onnx".format(version_num))

if __name__ == "__main__":
    version_t = (9, 11, 12, 13)
    for i in version_t:
        if i == 9 or i == 11:
            make_argmax(i)
        else:
            export_keepdims_select_last_index(i)