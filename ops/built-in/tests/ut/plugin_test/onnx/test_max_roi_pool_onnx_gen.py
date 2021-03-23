import onnx
from onnx import helper
from onnx import TensorProto


def make_max_roi_pool():
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 512, 19, 12])
    rois = helper.make_tensor_value_info('rois', TensorProto.FLOAT, [304, 5])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [304, 512, 7, 7])
    pooled_h, pooled_w = 7, 7
    spatial_value = 0.0635
    node_def = helper.make_node(
        'MaxRoiPool',
        inputs=['x', 'rois'],
        outputs=['y'],
        pooled_shape=[pooled_h, pooled_w],
        spatial_scale=spatial_value
    )

    graph = helper.make_graph(
        [node_def],
        'MaxRoiPool_v1',
        [x, rois],
        [y],
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 1
    onnx.save(model, "./test_max_roi_pool_case_1.onnx")
    onnx.checker.check_model(model)


if __name__ == '__main__':
    make_max_roi_pool()