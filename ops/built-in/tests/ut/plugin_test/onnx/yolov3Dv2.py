import onnx
from onnx import helper

def YoloV3DetectionOutputV2(version_num):
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 80, 32])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 32])
    z = helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1, 12, 32])
    box_out = helper.make_tensor_value_info("box_out", onnx.TensorProto.FLOAT, [1, 6, 512])
    box_out_num = helper.make_tensor_value_info("box_out_num", onnx.TensorProto.INT32, [1, 8])

    node = helper.make_node('YoloV3DetectionOutputV2',
                            inputs=['x', 'y', 'z'],
                            outputs=['box_out', 'box_out_num'],
                            N = 10,
                            boxes = 3,
                            coords = 4,
                            classes = 80,
                            post_nms_topn = 512,
                            pre_nms_topn = 512,
                            out_box_dim = 3,
                            obj_threshold = 0.5,
                            score_threshold = 0.5,
                            iou_threshold = 0.45,
                            biases = [12.0],
                            name='test_YoloV3DetectionOutputV2_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_YoloV3DetectionOutputV2_1",
        inputs=[x, y, z],
        outputs=[box_out, box_out_num]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_yolov3dv2_V{}.onnx".format(version_num))

if __name__ == '__main__':
    version_t = (8, 9, 10, 11, 12, 13)
    for i in version_t:
        YoloV3DetectionOutputV2(i)
