import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def roi_case(version_num):
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,1,10,10])
    rois  = helper.make_tensor_value_info('rois', TensorProto.FLOAT, [3,5])
    batch_indices =helper.make_tensor_value_info('batch_indices', TensorProto.INT32, [3])

    spatial_scale = helper.make_tensor_value_info('pooled_shape', AttributeProto.FLOAT, [])
    output_height = helper.make_tensor_value_info('pooled_shape', AttributeProto.INT, [])
    output_width = helper.make_tensor_value_info('pooled_shape', AttributeProto.INT, [])
    sampling_ratio = helper.make_tensor_value_info('pooled_shape', AttributeProto.INT, [])
    mode = helper.make_tensor_value_info('mode', AttributeProto.STRING, [])

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 5, 5])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'RoiAlign', # node name
        inputs=['X', 'rois', 'batch_indices'], # inputs
        outputs=['Y'], # outputs
        mode = 'avg',
        spatial_scale=1.0,
        output_height=5,
        output_width=5,
        sampling_ratio=2,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X,rois,batch_indices],
        [Y],
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='tr')
    model_def.opset_import[0].version = version_num
    onnx.save(model_def, "RoiAlign_V{}.onnx".format(version_num))
    print('The model is:\n{}'.format(model_def))

if __name__ == "__main__":
    version_t = (10, 11, 12, 13)
    for i in version_t:
        roi_case (i)
