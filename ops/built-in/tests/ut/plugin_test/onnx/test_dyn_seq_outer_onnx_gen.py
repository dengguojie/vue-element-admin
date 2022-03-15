import onnx
from onnx import helper
from onnx import TensorProto

def make_dyn_seq_outer(version):
    x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [-1])
    x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [-1])
    
    seq_len1 = helper.make_tensor_value_info('seq_len1', TensorProto.INT32, [1])
    seq_len2 = helper.make_tensor_value_info('seq_len2', TensorProto.INT32, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [-1])

    node_def = helper.make_node(
        'DynSeqOuter',
        inputs=['x1','x2','seq_len1','seq_len2'],
        outputs=['y'],
    )

    graph = helper.make_graph(
        [node_def],
        'test_dyn_seq_outer',
        inputs=[x1,x2,seq_len1,seq_len2],
        outputs=[y]
    )

    model = helper.make_model(graph, producer_name="onnx_parser_test")
    model.opset_import[0].version = version
    onnx.save(model, "./make_dyn_seq_outer.onnx")
    onnx.checker.check_model(model)



if __name__ == '__main__':
    make_dyn_seq_outer(8)
    make_dyn_seq_outer(9)
    make_dyn_seq_outer(10)
    make_dyn_seq_outer(11)
    make_dyn_seq_outer(12)
    make_dyn_seq_outer(13)

