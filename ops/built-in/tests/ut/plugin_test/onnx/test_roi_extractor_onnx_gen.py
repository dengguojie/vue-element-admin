import onnx
from onnx import helper


def make_roi_extractor_1():
    node = helper.make_node('RoiExtractor',
                            inputs=['features', 'rois'],
                            outputs=['y'],
                            name='test_roi_extractor_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_roi_extractor_1",
        inputs=[helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT16, [1, 16, 3, 4]),
                helper.make_tensor_value_info("rois", onnx.TensorProto.FLOAT16, [100, 5])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [100, 16, 7, 7])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_1")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_roi_extractor_case_1.onnx")


def make_roi_extractor_2():
    node = helper.make_node('RoiExtractor',
                            inputs=['features', 'rois'],
                            outputs=['y'],
                            pooled_height=7,
                            pooled_width=7,
                            name='test_roi_extractor_1')
    graph = helper.make_graph(
        nodes=[node],
        name="test_roi_extractor_1",
        inputs=[helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, [1, 256, 3, 4]),
                helper.make_tensor_value_info("rois", onnx.TensorProto.FLOAT, [100, 5])],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [100, 256, 7, 7])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test_2")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_roi_extractor_case_2.onnx")

if __name__ == '__main__':
    make_roi_extractor_1()
    make_roi_extractor_2()
