import onnx
from onnx import helper

def yolo(version_num):
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 255, 1, 1])
    coord_data = helper.make_tensor_value_info("coord_data", onnx.TensorProto.FLOAT, [1, 12, 17])
    obj_prob = helper.make_tensor_value_info("obj_prob", onnx.TensorProto.FLOAT, [1, 19])
    classes_prob = helper.make_tensor_value_info("classes_prob", onnx.TensorProto.FLOAT, [1, 80, 19])


    node = helper.make_node('Yolo',
                            inputs=['x'],
                            outputs=['coord_data', 'obj_prob', 'classes_prob'],
                            boxes=3,
                            coords=4,
                            classes=80,
                            yolo_version="V3",
                            name='test_yolo_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_yolo_1",
        inputs=[x],
        outputs=[coord_data, obj_prob, classes_prob]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_yolo_V{}.onnx".format(version_num))

if __name__ == '__main__':
    version_t = (8, 9, 10, 11, 12, 13)
    for i in version_t:
        yolo(i)
