import onnx
from onnx import helper


def make_col2im():
    node = helper.make_node('Col2im',
                            inputs=['A', 'B'],
                            outputs=['C'],
                            kernel_size=[3,3],
                            padding=[0,0],
                            stride=[1,1],
                            dilation=[1,1])
    graph = helper.make_graph(
        nodes=[node],
        name="test_col2im",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2,48,9,16]),
                helper.make_tensor_value_info("B", onnx.TensorProto.INT32, [2, ])],
        outputs=[helper.make_tensor_value_info(
            "C", onnx.TensorProto.FLOAT, [3, 1])]
    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_col2im_case_1.onnx")



if __name__ == '__main__':
    make_col2im()
