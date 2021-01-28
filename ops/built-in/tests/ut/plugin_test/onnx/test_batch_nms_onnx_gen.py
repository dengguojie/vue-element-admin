import onnx
from onnx import helper


def make_batch_nms_1():
    node = helper.make_node('BatchMultiClassNMS',
                            inputs=['boxes', 'scores'],
                            outputs=['Y1', 'Y2', 'Y3', 'Y4'],
                            iou_threshold=0.5,
                            score_threshold=0.0,
                            max_size_per_class=100,
                            max_total_size=100,
                            name='test_batch_nms_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_batch_nms_1",
        inputs=[helper.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT16, [1, 4741, 1, 4]),
                helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT16, [1, 4741, 1])],
        outputs=[helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT16, [1, 100, 4]),
                 helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT16, [1, 100]),
                 helper.make_tensor_value_info("Y3", onnx.TensorProto.FLOAT16, [1, 100]),
                 helper.make_tensor_value_info("Y4", onnx.TensorProto.INT32, [1])]
    )

    model = helper.make_model(graph, producer_name="onnx-batch_nms_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_batch_nms_1.onnx")


if __name__ == '__main__':
    make_batch_nms_1()
