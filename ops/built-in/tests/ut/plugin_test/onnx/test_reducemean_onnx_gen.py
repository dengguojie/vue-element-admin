import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_reducemean_1(version_num):
    X = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 2, 2])
    Y = helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [1, 1, 1])
    node_def = helper.make_node('ReduceMean',
                                inputs=['data'],
                                outputs=['reduced'],
                                keepdims=1,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_reducemean_case_1",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-reducemean_test")
    
    if version_num == 11:
        model.opset_import[0].version = version_num
        onnx.save(model, "./test_reducemean_case_1.onnx")
    elif version_num == 13:
        model.opset_import[0].version = version_num
        onnx.save(model, "./ReduceMean_default_V13.onnx")
    elif version_num == 12:
        model.opset_import[0].version = version_num
        onnx.save(model, "./ReduceMean_default_V12.onnx")
    else version_num == 9:
        model.opset_import[0].version = version_num
        onnx.save(model, "./ReduceMean_default_V9.onnx")


def make_reducemean_2(version_num):
    X = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3, 2])
    Y = helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [4, 2])
    node_def = helper.make_node('ReduceMean',
                                inputs=['data'],
                                outputs=['reduced'],
                                keepdims=0,
                                axes=[1]
                                )
    graph = helper.make_graph(
        [node_def],
        "test_reducemean_case_2",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-reducemean_test")
    model.opset_import[0].version = version_num
    onnx.save(model, "./test_reducemean_case_2.onnx")


def reducemean_do_not_keepdims(version_num):
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 2])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'ReduceMean',
        inputs=['input'],
        outputs=['output'],
        axes = [1],
        keepdims = 0
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceMean-onnx')
    
    if version_num == 12:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_not_keepdims_V12.onnx")
    elif version_num == 13:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_not_keepdims_V13.onnx")
    else version_num == 9:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_not_keepdims_V9.onnx")


def reducemean_keepdims(version_num):
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 1, 2])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'ReduceMean',
        inputs=['input'],
        outputs=['output'],
        axes = [1],
        keepdims = 1
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceMean-onnx')
    
    if version_num == 12:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_keepdims_V12.onnx")
    elif version_num == 13:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_keepdims_V13.onnx")
    else version_num == 9:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_keepdims_V9.onnx")


def reducemean_negative_axes_keepdims(version_num):
    #Create one input (ValueInfoProto)
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 2])

    #Create one output (ValueOutputProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 1, 2])

    # Create a node (NodeProto)
    node_def = onnx.helper.make_node(
        'ReduceMean',
        inputs=['input'],
        outputs=['output'],
        axes = [-2],
        keepdims = 1
        )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input],
        [output]
        )
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='zyw-ReduceMean-onnx')
    
    if version_num == 12:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_negative_V12.onnx")
    elif version_num == 13:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_negative_V13.onnx")
    else version_num == 9:
        model.opset_import[0].version = version_num
        onnx.save(model_def, "./ReduceMean_negative_V9.onnx")


if __name__ == '__main__':
    version_list = "9 11 12 13"
    for i in $version_list; do
        if version_num == 11:
            make_reducemean_1(version_num)
            make_reducemean_2(version_num)
        elif version_num == 13:
            reducemean_do_not_keepdims(version_num)
            reducemean_keepdims(version_num)
            reducemean_negative_axes_keepdims(version_num)
            make_reducemean_1(version_num)
        elif version_num == 12:
            reducemean_do_not_keepdims(version_num)
            reducemean_keepdims(version_num)
            reducemean_negative_axes_keepdims(version_num)
            make_reducemean_1(version_num)
        else :
            reducemean_do_not_keepdims(version_num)
            reducemean_keepdims(version_num)
            reducemean_negative_axes_keepdims(version_num)
            make_reducemean_1(version_num)