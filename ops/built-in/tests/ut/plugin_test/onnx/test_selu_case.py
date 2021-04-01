import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def Selu(version_num):
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 4, 5])

    node_def = onnx.helper.make_node(
        'Selu',
        inputs=['input'],
        outputs=['output']
        )

    graph_def = helper.make_graph(
        [node_def],
        'Selu',
        [x],
        [y]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='selu-onnx')
    model_def.opset_import[0].version = version_num

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_selu_case_{}.onnx".format(version_num))
    print('The model is:\n{}'.format(model_def))

def SeluAttr():
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 4, 5])

    node_def = onnx.helper.make_node(
        'Selu',
        inputs=['input'],
        outputs=['output'],
        alpha=2.0,
        gamma=3.0
        )

    graph_def = helper.make_graph(
        [node_def],
        'Selu',
        [x],
        [y]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='selu-onnx')
    model_def.opset_import[0].version = 11

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "./test_selu_case_with_attr.onnx")
    print('The model is:\n{}'.format(model_def))


if __name__ == "__main__":
    versions = [9, 10, 11, 12, 13]
    for i in versions:
        Selu(i)
    SeluAttr()
