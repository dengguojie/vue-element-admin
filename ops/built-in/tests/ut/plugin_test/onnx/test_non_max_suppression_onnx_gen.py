import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np


def make_non_max_suppression_input5():
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1,6,4])
    # Create one output (ValueInfoProto)
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1,1,6])
    max_output_boxes_per_class = helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [2])
    iou_threshold = helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [0.7])
    score_threshold = helper.make_tensor('score_threshold', TensorProto.FLOAT, [1], [0.0])
    # Create one output (ValueInfoProto)
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [2,3])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices'],
        center_point_box=0
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [boxes,scores],
        [selected_indices],
        [max_output_boxes_per_class,iou_threshold,score_threshold]
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx_parser_test')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_make_non_max_suppression_input5_case.onnx")


def make_non_max_suppression_input4():
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1,6,4])
    # Create one output (ValueInfoProto)
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1,1,6])
    max_output_boxes_per_class = helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [2])
    iou_threshold = helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [0.7])
    # Create one output (ValueInfoProto)
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [2,3])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold'],
        outputs=['selected_indices'],
        center_point_box=0
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [boxes,scores],
        [selected_indices],
        [max_output_boxes_per_class,iou_threshold]
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx_parser_test')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_make_non_max_suppression_input4_case.onnx")


def make_non_max_suppression_input3():
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1,6,4])
    # Create one output (ValueInfoProto)
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1,1,6])
    max_output_boxes_per_class = helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [2])
    # Create one output (ValueInfoProto)
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [2,3])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class'],
        outputs=['selected_indices'],
        center_point_box=0
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [boxes,scores],
        [selected_indices],
        [max_output_boxes_per_class]
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx_parser_test')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_make_non_max_suppression_input3_case.onnx")


if __name__ == '__main__':
    make_non_max_suppression_input5()
    make_non_max_suppression_input4()
    make_non_max_suppression_input3()
