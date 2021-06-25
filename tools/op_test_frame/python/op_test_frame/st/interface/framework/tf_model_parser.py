#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function:
 TFModelParse class. This class mainly get case info from tf model.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
Change History: 2020-07-11 file Created
"""
try:
    import os
    import sys
    import tensorflow as tf
    from tensorflow.core.framework import tensor_shape_pb2
    from .. import utils
except (ImportError,) as import_error:
    sys.exit(
        "[tf_model_parser] Unable to import module: %s." % str(import_error))


TMP_SHAPE_FILE = 'tmp_shape.json'
TMP_GA_PATH_FILE = 'tmp_ga_path.json'


def _map_tf_input_output_dtype(tf_dtype):
    op_dtype = "UNKNOW"
    if tf_dtype == 'float32':
        op_dtype = "float"
    elif tf_dtype == 'float16':
        op_dtype = "float16"
    elif tf_dtype == 'int8':
        op_dtype = "int8"
    elif tf_dtype == 'int16':
        op_dtype = "int16"
    elif tf_dtype == 'int32':
        op_dtype = "int32"
    elif tf_dtype == 'uint8':
        op_dtype = "uint8"
    elif tf_dtype == 'uint16':
        op_dtype = "uint16"
    elif tf_dtype == 'uint32':
        op_dtype = "uint32"
    elif tf_dtype == 'bool':
        op_dtype = "bool"
    else:
        utils.print_warn_log("The op type: %s is unsupported. Please check." %
                             tf_dtype)
    return op_dtype


def _attr_value_shape_list(attr_value_shape):
    attr_list = []
    if not attr_value_shape:
        return attr_list
    for i in attr_value_shape:
        attr_list.append(i.size)
    return attr_list


def _tensor_shape_list(tensor_shape):
    tensor_list = []
    if not tensor_shape:
        return tensor_list
    for i in tensor_shape:
        if isinstance(i, tf.TensorShape):
            tensor_list.append(i.as_list())
        else:
            tensor_list.append([])
    return tensor_list


def _parse_attr(attr_key, value_data):
    attr = {'name': attr_key,
            'type': "",
            'value': []}
    if value_data.list.shape:
        attr['type'] = "list(shape)"
        for value_sharp in value_data.list.shape:
            attr['value'] += _attr_value_shape_list(value_sharp.dim)
    elif value_data.shape.dim:
        attr['type'] = "shape"
        attr['value'] += _attr_value_shape_list(value_data.shape.dim)
    elif value_data.list.s:
        attr['type'] = "list(string)"
        attr['value'] = "".join(value_x.decode("utf-8")
                                for value_x in value_data.list.s)
    elif value_data.list.i:
        attr['type'] = "list(int)"
        attr['value'] += value_data.list.i
    elif value_data.list.f:
        attr['type'] = "list(float)"
        attr['value'] += value_data.list.f
    else:
        value_data = str(value_data).replace('\n', utils.EMPTY).split(":")
        if len(value_data) == 2:
            attr['type'] = value_data[0]
            attr['value'] = value_data[1]
    return attr


def _get_node_attr(node, key):
    try:
        return node.attr.get(key)
    except KeyError as ex:
        utils.print_error_log(
            'Failed to load tf graph . %s' % (str(ex)))
        raise utils.OpTestGenException(
            utils.OP_TEST_GEN_GET_KEY_ERROR)


def _tf_utils_load_graph_def(filename):
    # load model
    try:
        with tf.io.gfile.GFile(filename, "rb") as graph_file:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(graph_file.read())
        return graph_def
    except Exception as ex:
        utils.print_error_log(
            'Failed to load tf graph "%s". %s' % (filename, str(ex)))
        raise utils.OpTestGenException(
            utils.OP_TEST_GEN_TF_LOAD_ERROR)


def _tf_utils_get_operators(graph_def):
    # get operators
    try:
        tf.import_graph_def(graph_def, name='')
        all_ops = tf.compat.v1.get_default_graph().get_operations()
        return all_ops
    except Exception as ex:
        utils.print_error_log(
            'Failed to get operators. %s' % (str(ex)))
        raise utils.OpTestGenException(
            utils.OP_TEST_GEN_TF_GET_OPERATORS_ERROR)


def _tf_utils_write_graph(graph_def, dir_name, new_graph_path):
    # modify tf graph
    try:
        tf.io.write_graph(graph_def, dir_name, new_graph_path,
                          as_text=False)
    except Exception as ex:
        utils.print_error_log(
            'Failed to write new graph file %s. %s' % (new_graph_path,
                                                       str(ex)))
        raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)


def _get_node_lists_match_ini_op(i, op_type, node_info, ops):
    nodes_list = list()
    op_infos = {"op_type": op_type, "layer": node_info.name,
                "input_dtype": [],
                "output_dtype": [], "input_shape": [],
                "output_shape": [], "attr": []}
    # input_shape
    input_shape = _tensor_shape_list(ops[i]['input_shape'])
    op_infos["input_shape"] = input_shape
    # input_dtype
    input_dtype = [_map_tf_input_output_dtype(x.name) for x in
                   ops[i]['input_dtype']]
    op_infos["input_dtype"] = input_dtype
    # output_shape
    output_shape = _tensor_shape_list(ops[i]['output_shape'])
    op_infos["output_shape"] = output_shape
    # output_dtype
    output_dtype = [_map_tf_input_output_dtype(x.name) for x in
                    ops[i]['output_dtype']]
    op_infos["output_dtype"] = output_dtype

    # attr
    for attr_key in node_info.attr.keys():
        value_data = node_info.attr[attr_key]
        attr = _parse_attr(attr_key, value_data)
        op_infos["attr"].append(attr)
    nodes_list.append(op_infos)
    return nodes_list


def _get_nodes_list(model_path, ini_op_type):
    nodes_list = list()
    graph_def = _tf_utils_load_graph_def(os.path.realpath(model_path))
    all_ops = _tf_utils_get_operators(graph_def)
    ops = []
    for operator in all_ops:
        op_info = {"name": operator.node_def.name, "input_dtype": [],
                   "output_dtype": [], "input_shape": [],
                   "output_shape": []}
        for input_tensor in operator.inputs:
            op_info["input_shape"].append(input_tensor.get_shape())
            op_info["input_dtype"].append(input_tensor.dtype)
        for out_tensor in operator.outputs:
            op_info["output_shape"].append(out_tensor.get_shape())
            op_info["output_dtype"].append(out_tensor.dtype)
        ops.append(op_info)
    for i, node_info in enumerate(graph_def.node):
        op_type = node_info.op
        if op_type == ini_op_type:
            nodes_list = _get_node_lists_match_ini_op(
                i, op_type, node_info, ops)
    return nodes_list


class TFModelParse:
    """
    the class for parse tf model.
    """

    def __init__(self, args):
        if hasattr(args, 'input_file'):
            self.input_file = args.input_file
        self.output_path = args.output_path
        self.op_info = {}
        self.op_name = ''
        if hasattr(args, 'model_path'):
            self.model_path = args.model_path
        if hasattr(args, 'quiet'):
            self.quiet_flag = args.quiet

    def _change_shape_fn(self, file_path, new_shape_map):
        real_path = os.path.realpath(file_path)
        graph_def = _tf_utils_load_graph_def(real_path)
        node_shape = ""
        for node in graph_def.node:
            if node.op == "Placeholder" and node.name in new_shape_map \
                    and 'new_shape' in new_shape_map[node.name] \
                    and new_shape_map[node.name]['new_shape']:
                new_shape = list()
                for dim in new_shape_map[node.name]['new_shape']:
                    new_shape.append(
                        tensor_shape_pb2.TensorShapeProto.Dim(size=int(dim)))
                node.attr['shape'].shape.CopyFrom(
                    tensor_shape_pb2.TensorShapeProto(dim=new_shape))
                node_shape = _get_node_attr(node, 'shape')
                utils.print_info_log("The %s shape has been changed "
                                     "to %s " % (node.name, node_shape))
        if not node_shape:
            utils.print_error_log("Change shape failed, maybe there is no "
                                  "match layer. Please check the input shape.")
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR)
        dir_name, tmp_filename = os.path.split(real_path)
        prefix, _ = os.path.splitext(tmp_filename)
        first_new_shape = '_'.join(
            str(i) for i in list(new_shape_map.values())[0]['new_shape'])
        new_graph_path = os.path.realpath(
            os.path.join(self.output_path, prefix + "_%s.pb" % first_new_shape))
        _tf_utils_write_graph(graph_def, dir_name, new_graph_path)
        return node_shape, new_graph_path

    @staticmethod
    def _check_ori_shape_and_notice(shape, layer_name):
        if isinstance(shape, list):
            for dim in shape:
                if not isinstance(dim, int):
                    utils.print_warn_log(
                        'The input shape(%s) of layer(%s) is not a number. '
                        'Try to change the "placeholder" shape to fix the '
                        'problem.' % (shape, layer_name))
                    return
                if dim <= 0:
                    utils.print_warn_log(
                        'The input shape(%s) of layer(%s) must be greater than'
                        ' 0. Try to change the "placeholder" shape '
                        'to fix the problem.' % (shape, layer_name))
                    return
        else:
            utils.print_warn_log(
                'The input shape(%s) format error.Please retype.' % shape)
        return

    def _get_placeholder_shape_from_user(self, shape_map):
        result = False
        for (layer_name, value) in shape_map.items():
            node_shape = value["ori_shape"]
            self._check_ori_shape_and_notice(node_shape, layer_name)
            utils.print_info_log('"%s" layer is a "placeholder" '
                                 'operator , the original input shape : %s'
                                 % (layer_name, node_shape))
            while True:
                new_placeholder_shape = input(
                    'Would you like to change the "placeholder" shape ? '
                    'If yes, please enter a new shape like 8,224,224,3.'
                    'if not, enter "n" skip:')
                if new_placeholder_shape.lower() == 'n':
                    utils.print_info_log("Skip change shape.")
                    value["new_shape"] = node_shape
                    break
                if self._check_new_shape_and_notice(new_placeholder_shape):
                    value["new_shape"] = new_placeholder_shape.split(',')
                    result = True
                    break
                utils.print_warn_log(
                    "The input shape above is invalid, please retype!")
        return result

    def get_tf_model_nodes(self, op_type):
        """
        get layers information from tf model
        :param op_type: op_type from ini file, eg:"Add"
        :return: list store the layer information
        """
        self._check_get_nodes_argument_valid(op_type)
        if not self.quiet_flag:
            shape_map = self._get_shape_fn(self.model_path)
            if self._get_placeholder_shape_from_user(shape_map):
                _, file_path = self._change_shape_fn(self.model_path,
                                                     shape_map)
            else:
                file_path = self.model_path
            utils.print_info_log("Begin to get the \"%s\" operator in the "
                                 "model %s." % (op_type, file_path))
            nodes_list = _get_nodes_list(file_path, op_type)
        else:
            nodes_list = _get_nodes_list(self.model_path, op_type)
        return nodes_list

    @staticmethod
    def _get_shape_fn(file_path):
        graph_def = _tf_utils_load_graph_def(file_path)
        placeholder_shape_map = {}
        for node in graph_def.node:
            if node.op == "Placeholder":
                node_shape = _get_node_attr(node, 'shape')
                layer_name = node.name
                value = _attr_value_shape_list(node_shape.shape.dim)
                if not value or not layer_name:
                    utils.print_warn_log(
                        "The layer(%s) is \"placeholder\", the shape is "
                        "empty!" % layer_name)
                else:
                    placeholder_shape_map.update(
                        {layer_name: {'ori_shape': value, 'new_shape': []}})
        if not placeholder_shape_map:
            utils.print_error_log("There is no \"placeholder\" operator !"
                                  "Please check the model.")
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_TF_GET_PLACEHOLDER_ERROR)
        return placeholder_shape_map

    @staticmethod
    def _check_new_shape_and_notice(shape):
        if isinstance(shape, str):
            shape = shape.split(',')
            for dim in shape:
                if not dim.isdigit():
                    utils.print_warn_log(
                        'The dim(%s) in shape(%s) is not a number.'
                        % (dim, shape))
                    return False
                if int(dim) <= 0:
                    utils.print_warn_log(
                        'The dim(%s) in shape(%s) must be greater than 0.'
                        % (dim, shape))
                    return False
            return True
        utils.print_warn_log(
            'The input shape(%s) format error.Please retype.' % shape)
        return False

    def get_shape(self):
        """
        interface for IDE , get "Placeholder" shape from tf model
        generate the json file with the shape store in
        """
        self._check_get_shape_argument_valid()
        shape_map = self._get_shape_fn(self.model_path)
        json_path = os.path.join(self.output_path, TMP_SHAPE_FILE)
        utils.write_json_file(json_path, shape_map)

    def _check_get_nodes_argument_valid(self, op_type):
        utils.check_name_valid(op_type)
        utils.check_path_valid(self.model_path)
        utils.check_path_valid(os.path.realpath(self.output_path), True)

    def _check_get_shape_argument_valid(self):
        utils.check_path_valid(self.model_path)
        utils.check_path_valid(os.path.realpath(self.output_path), True)

    def _check_change_shape_argument_valid(self):
        if not self.input_file.endswith(".json"):
            utils.print_error_log(
                'The file "%s" is invalid, only supports .json file. '
                'Please modify it.' % self.input_file)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_PATH_ERROR)
        utils.check_path_valid(self.input_file)
        utils.check_path_valid(self.model_path)
        utils.check_path_valid(os.path.realpath(self.output_path), True)

    def change_shape(self):
        """
        interface for IDE , change "Placeholder" shape from tf model
        generate the json file with the new model path store in
        """
        self._check_change_shape_argument_valid()
        new_shape_map = utils.load_json_file(self.input_file)
        _, new_model_path = self._change_shape_fn(self.model_path,
                                                  new_shape_map)
        json_path = os.path.realpath(os.path.join(self.output_path,
                                                  TMP_GA_PATH_FILE))
        utils.write_json_file(json_path, {'new_model_path': new_model_path})


def get_model_nodes(args, op_type):
    """
    get layer information  which op type is op_type from tf model
    :param args: the args config by user
    :param op_type: op_type from ini file, eg:"Add"
    :return: node list to store the layer information
    """
    tf_parser = TFModelParse(args)
    return tf_parser.get_tf_model_nodes(op_type)


def get_shape(args, op_type):
    """
    interface for IDE , get "Placeholder" shape from tf model
    :return: the json file with the shape store in
    """
    _ = op_type
    tf_parser = TFModelParse(args)
    return tf_parser.get_shape()


def change_shape(args, op_type):
    """
    interface for IDE , change "Placeholder" shape from tf model
    :return: the json file with the new model path store in
    """
    _ = op_type
    tf_parser = TFModelParse(args)
    return tf_parser.change_shape()
