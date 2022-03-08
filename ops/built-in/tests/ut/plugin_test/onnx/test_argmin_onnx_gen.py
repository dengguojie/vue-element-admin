#argmax.py
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def export_keepdims_select_last_index(version_num):
    data = helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 2])
    result = helper.make_tensor_value_info("result", onnx.TensorProto.INT64, [2, 1])

    node = helper.make_node(
        'ArgMin',
        inputs=['data'],
        outputs=['result'],
        axis=1,
        keepdims=1,
    )

    graph_def = helper.make_graph(
        [node],
        'test-model',
        [data],
        [result],
    )

    model_def = onnx.helper.make_model(graph_def, producer_name='HJ-ArgMin-onnx')
    model_def.opset_import[0].version = version_num
    onnx.save(model_def, "./test_argmin_case_V{}.onnx".format(version_num))

if __name__ == "__main__":
    i = 13
    export_keepdims_select_last_index(i)
