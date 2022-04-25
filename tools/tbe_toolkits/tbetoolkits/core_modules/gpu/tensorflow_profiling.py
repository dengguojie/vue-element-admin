#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Profiling method for Pytorch
"""
# Standard Packages
import inspect
import json
import logging
import os
from typing import Sequence

# Third-party Packages
import numpy as np

from .input_generation import __gen_input
from ..tbe_multiprocessing.pool import get_process_context
from ...utilities import get_tf_func
from ...utilities import apply_as_list
from ...utilities import get_global_storage
from ...utilities import set_thread_name
from ...utilities import set_process_name
from ...utilities import eliminate_scalar_shapes
from ..testcase_manager import UniversalTestcaseStructure


def __notify_status(status: str):
    set_thread_name(status)
    get_process_context().send_data("stage", status)


def __report_process_name(my_name_is: str):
    set_process_name(my_name_is)
    get_process_context().change_name(my_name_is)


def tensorflow_profiling(context: UniversalTestcaseStructure, device_id):
    # Do parameter mapping
    from ...user_defined_modules.davinci_to_tf import dav2tf_registry
    if context.op_name in dav2tf_registry.dav_op_to_tf_map:
        context = dav2tf_registry.dav_op_to_tf_map[context.op_name](context)
    op_name = context.op_name
    testcase_name = context.testcase_name
    __notify_status("OnLoadTensorflow")
    tf = __import__("tensorflow")
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
    tf_func = get_tf_func(op_name)
    if tf_func is None:
        logging.error("\n===========================\n"
                      "Operator %s not found in tensorflow\n"
                      "===========================\n" % op_name)
        return "UNSUPPORTED"
    if "SHAPE_OUT_OF_BOUND" in str(context.original_line):
        return "SHAPE_OUT_OF_BOUND"
    if "OUTPUT_INFERENCE_FAILED" in str(context.original_line):
        return "OUTPUT_INFERENCE_FAILED"
    __notify_status("OnGenInput")
    feed_dict = {}
    __gen_input(context)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    __notify_status("OnCompilation")
    stc_op, input_map = \
        __compile_tf_func(tf_func, context, 0, get_global_storage().run_time)
    feed_dict.update(input_map)
    sess_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                           inter_op_parallelism_threads=1,
                                           gpu_options=tf.compat.v1.GPUOptions(force_gpu_compatible=True,
                                                                               allow_growth=True))
    __notify_status("OnTFSession")
    profiling_results = {"static": {}, "jsons": []}
    with tf.compat.v1.Session(config=sess_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.HARDWARE_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        sess.run([stc_op],
                 options=run_options,
                 run_metadata=run_metadata,
                 feed_dict=feed_dict)
        __parse_tf_profiling_result(profiling_results["static"], run_metadata, testcase_name, profiling_results)
    __notify_status("OnJsonWrite")
    for idx, _json in enumerate(profiling_results["jsons"]):
        with open("gpu_json/%s_run%d.json" % (testcase_name, idx), "w+") as f:
            f.write(json.dumps(_json, indent=4))
    __notify_status("OnReturn")
    perf = sum(tuple(profiling_results["static"].values())) / get_global_storage().run_time
    if np.isnan(perf) or perf <= 0:
        raise RuntimeError("Perf result unavailable")
    return perf


def __compile_tf_func(tf_func, context: UniversalTestcaseStructure, device, run_time):
    import tensorflow as tf
    kernel_name = "%s_dev%d" % (context.testcase_name, device)
    original_params = {**context.other_runtime_params, **context.other_compilation_params}
    params = original_params.copy()
    funcs = []
    input_map = {}
    with tf.compat.v1.device("/device:GPU:%d" % device):
        with tf.compat.v1.variable_scope("%s" % context.testcase_name, reuse=tf.compat.v1.AUTO_REUSE):
            for i in range(run_time):
                stc_inputs = [tf.compat.v1.placeholder(shape=context.input_arrays[j].shape,
                                                       dtype=context.input_arrays[j].dtype)
                              if j not in context.input_as_variable else tf.compat.v1.get_variable(
                    name="temp_stc_var_%d_run%d" % (j, i),
                    shape=context.input_arrays[j].shape,
                    dtype=context.input_arrays[j].dtype,
                    initializer=tf.compat.v1.random_normal_initializer())
                              for j in range(len(context.input_arrays)) if context.input_arrays[j] is not None]
                mappable_stc_inputs = [ipt for ipt in stc_inputs if not isinstance(ipt, tf.Variable)]
                mappable_inputs = [context.input_arrays[j] for j in range(len(context.input_arrays))
                                   if context.input_arrays[j] is not None and j not in context.input_as_variable]
                input_map.update(dict(zip(mappable_stc_inputs, mappable_inputs)))
                # input_as_list
                stc_inputs = apply_as_list(stc_inputs, context.stc_input_as_list_distribution)
                # Remove useless keys
                tf_func_params = tuple(inspect.signature(tf_func).parameters.keys())
                increased_index = 0
                for idx, value in enumerate(stc_inputs):
                    if idx + increased_index >= len(tf_func_params):
                        break
                    while tf_func_params[idx + increased_index] in params:
                        increased_index += 1
                    else:
                        params[tf_func_params[idx + increased_index]] = value
                if "name" in params:
                    params["name"] = "%s_run%d" % (kernel_name, i)
                for param in tuple(params.keys()):
                    if param not in tf_func_params:
                        logging.warning("Tensorflow function compilation removing redundant parameter: %s" % param)
                        del params[param]
                stc_func = tf_func(**params)
                while isinstance(stc_func, Sequence):
                    stc_func = stc_func[0]
                params = original_params.copy()
                funcs.append(tf.reshape(stc_func, eliminate_scalar_shapes((stc_func.shape,))[0]))
            stc_func = tf.concat(funcs, axis=0, name="stc_final")
    return stc_func, input_map


def __parse_tf_profiling_result(profiling_results, run_metadata, testcase_name, full_results):
    for dev_stat in run_metadata.step_stats.dev_stats:
        if dev_stat.device.endswith("all"):
            for node_stat in dev_stat.node_stats:
                node_name = node_stat.node_name.split(":")[0]
                if node_name == "Thunk":
                    # XLA Mode
                    node_name = node_stat.timeline_label.split("@@")[1].split(":")[0]
                if node_name.startswith("dyn_%s" % testcase_name):
                    logging.debug(
                        "Dynamic result received: %s with %s" % (node_stat.node_name, node_stat.op_end_rel_micros))
                    profiling_results.setdefault(node_name, 0)
                    profiling_results[node_name] += node_stat.op_end_rel_micros
                elif node_name.startswith("stc_%s" % testcase_name):
                    logging.debug(
                        "Static result received: %s with %s" % (node_stat.node_name, node_stat.op_end_rel_micros))
                    profiling_results.setdefault(node_name, 0)
                    profiling_results[node_name] += node_stat.op_end_rel_micros
                elif node_name.startswith(testcase_name):
                    logging.debug(
                        "Unknown result received: %s with %s" % (node_stat.node_name, node_stat.op_end_rel_micros))
                    profiling_results.setdefault(node_name, 0)
                    profiling_results[node_name] += node_stat.op_end_rel_micros
                elif "xla_run" in node_name:
                    logging.debug(
                        "Xla result received: %s with %s" % (node_stat.node_name, node_stat.op_end_rel_micros))
                    profiling_results.setdefault(node_name, 0)
                    profiling_results[node_name] += node_stat.op_end_rel_micros
    # noinspection PyUnresolvedReferences
    from tensorflow.python.client import timeline
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = json.loads(tl.generate_chrome_trace_format())
    full_results["jsons"].append(ctf)
