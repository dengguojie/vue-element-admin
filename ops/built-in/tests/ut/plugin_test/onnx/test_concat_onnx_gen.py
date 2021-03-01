import onnx
from onnx import helper


def make_concat(version_num):
    node = helper.make_node('Concat',
                            inputs=['A'],
                            outputs=['C'],
                            axis=0,
                            name='test_concat')
    graph = helper.make_graph(
        nodes=[node],
        name="test_concat",
        inputs=[helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [2, 3, 4])]

    )

    model = helper.make_model(graph, producer_name="onnx-parser_test")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_concat_case_V{}.onnx".format(version_num))
    onnx.checker.check_model(model)


if __name__ == '__main__':
    concat_list = (9, 11, 12, 13)
    for i in concat_list:
        make_concat(i)


    
