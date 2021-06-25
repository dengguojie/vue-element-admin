import onnx
from onnx import helper
import numpy as np

def make_split_v11_fail():
    node = helper.make_node(
        'Split',
        inputs=['input'],
        outputs=['output_0', 'output_1', 'output_2'],
        axis=3,
        split=[1, 1],
        name='split_v11'
    )
    graph = helper.make_graph(
        nodes=[node],
        name='split_v11_no_default',
        inputs=[helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 2, 3, 4])],
        outputs=[helper.make_tensor_value_info('output_0', onnx.TensorProto.FLOAT, [1, 2, 3, 1]),
                 helper.make_tensor_value_info('output_1', onnx.TensorProto.FLOAT, [1, 2, 3, 1]),
                 helper.make_tensor_value_info('output_2', onnx.TensorProto.FLOAT, [1, 2, 3, 2])]
    )

    model = helper.make_model(graph, producer_name='onnx')
    model.opset_import[0].version = 11
    onnx.save(model, './test_split_v11_no_default.onnx')


def make_split_v11_default_split():
    node = helper.make_node(
        'Split',
        inputs=['input'],
        outputs=['output_0', 'output_1', 'output_2'],
        axis=3,
        name='split_v11'
    )
    graph = helper.make_graph(
        nodes=[node],
        name='split_v11_default_split',
        inputs=[helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 2, 3, 6]),],
        outputs=[helper.make_tensor_value_info('output_0', onnx.TensorProto.FLOAT, [1, 2, 3, 2]),
                 helper.make_tensor_value_info('output_1', onnx.TensorProto.FLOAT, [1, 2, 3, 2]),
                 helper.make_tensor_value_info('output_2', onnx.TensorProto.FLOAT, [1, 2, 3, 2])]
    )

    model = helper.make_model(graph, producer_name='onnx')
    model.opset_import[0].version = 11
    onnx.save(model, './test_split_v11_default_split.onnx')

def make_split_v13_no_default():
    node = helper.make_node(
        'Split',
        inputs=['input', 'split'],
        outputs=['output_0', 'output_1', 'output_2'],
        axis=3,
        name='split_v13'
    )
    const_tensor = np.array([1, 2, 3]).astype(np.int64).flatten().tolist()
    graph = helper.make_graph(
        nodes=[node],
        name='split_v13_no_default',
        inputs=[helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 2, 3, 6]),
                helper.make_tensor_value_info('split', onnx.TensorProto.INT64, [3, ])],
        outputs=[helper.make_tensor_value_info('output_0', onnx.TensorProto.FLOAT, [1, 2, 3, 1]),
                 helper.make_tensor_value_info('output_1', onnx.TensorProto.FLOAT, [1, 2, 3, 2]),
                 helper.make_tensor_value_info('output_2', onnx.TensorProto.FLOAT, [1, 2, 3, 3])]
    )

    model = helper.make_model(graph, producer_name='onnx')
    model.opset_import[0].version = 13
    onnx.save(model, './test_split_v13_no_default.onnx')

def make_split_v11_split():
    node = helper.make_node(
        'Split',
        inputs=['input'],
        outputs=['output_0', 'output_1', 'output_2'],
        axis=3,
        split=[1, 1, 4],
        name='split_v11'
    )
    graph = helper.make_graph(
        nodes=[node],
        name='split_v11_default_split',
        inputs=[helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 2, 3, 6]),],
        outputs=[helper.make_tensor_value_info('output_0', onnx.TensorProto.FLOAT, [1, 2, 3, 2]),
                 helper.make_tensor_value_info('output_1', onnx.TensorProto.FLOAT, [1, 2, 3, 2]),
                 helper.make_tensor_value_info('output_2', onnx.TensorProto.FLOAT, [1, 2, 3, 2])]
    )

    model = helper.make_model(graph, producer_name='onnx')
    model.opset_import[0].version = 11
    onnx.save(model, './test_split_v11_split.onnx')

if __name__ == '__main__':
    make_split_v11_fail()
    make_split_v11_default_split()
    make_split_v13_no_default()
    make_split_v11_split()

