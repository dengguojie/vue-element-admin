import onnx
from onnx import helper

def yolov5(version_num):
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 255, 1, 1])
    coord_data = helper.make_tensor_value_info("coord_data", onnx.TensorProto.FLOAT, [1, 12, 17])
    obj_prob = helper.make_tensor_value_info("obj_prob", onnx.TensorProto.FLOAT, [1, 19])


    node = helper.make_node('YoloV5DetectionOutput',
                            inputs=['x'],
                            outputs=['coord_data', 'obj_prob'],
                            N = 2,
                            biases = [0.4, 0.5],
                            boxes=3,
                            coords=4,
                            classes=80,
                            relative = 1,
                            post_nms_topn = 512,
                            pre_nms_topn = 512,
                            out_box_dim = 3,
                            obj_threshold = 0.5,
                            score_threshold = 0.5,
                            iou_threshold = 0.45,
                            alpha = 2.0,
                            resize_origin_img_to_net = 0,
                            name='test_yolo_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_yolo_1",
        inputs=[x],
        outputs=[coord_data, obj_prob]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_yolov5_V{}.onnx".format(version_num))

def yolov5_fail(version_num):
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 255, 1, 1])
    coord_data = helper.make_tensor_value_info("coord_data", onnx.TensorProto.FLOAT, [1, 12, 17])
    obj_prob = helper.make_tensor_value_info("obj_prob", onnx.TensorProto.FLOAT, [1, 19])


    node = helper.make_node('YoloV5DetectionOutput',
                            inputs=['x'],
                            outputs=['coord_data', 'obj_prob'],
                            N = 2,
                            boxes=3,
                            coords=4,
                            classes=80,
                            relative = 1,
                            post_nms_topn = 512,
                            pre_nms_topn = 512,
                            out_box_dim = 3,
                            obj_threshold = 0.5,
                            score_threshold = 0.5,
                            iou_threshold = 0.45,
                            alpha = 2.0,
                            resize_origin_img_to_net = 0,
                            name='test_yolo_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_yolo_1",
        inputs=[x],
        outputs=[coord_data, obj_prob]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_yolov5_V{}_fail.onnx".format(version_num))

if __name__ == '__main__':
    yolov5(11)
    yolov5_fail(11)
