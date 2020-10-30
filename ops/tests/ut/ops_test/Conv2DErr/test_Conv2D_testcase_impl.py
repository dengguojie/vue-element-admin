# pylint: disable=too-many-statements, too-many-locals,superfluous-parens,broad-except
# pylint: disable=too-many-branches, too-many-arguments
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_conv_cce(test_arg):
  import sys
  import impl
  sys.path.append("./llt/tensor_engine/ut/testcase_python")
  from te import tvm
  from impl.conv2d import calc_para_from_dict
  from impl.conv2d import op_select_format
  from impl.conv2d import _conv_layer_cce
  from impl.conv2d import conv2d_compute
  from impl.conv2d_compress import conv2dcompress_compute
  from impl.conv2d_compress import _conv_layer_compress_cce
  from te import platform as cceconf

  succ_str = "OK"
  fail_str = "FAILED"

  def verify_conv2d_calc_para_from_dict(inputs, weights, strides, pads, dilations, outputs):
    calc_para_from_dict(inputs, weights, strides, pads, dilations, outputs)

  def verify_conv2d_op_select_format(inputs, weights, bias, offset_w, outputs, strides,
                      pads, dilations):
      op_select_format(inputs, weights, bias, offset_w, outputs, strides,
                      pads, dilations)
  def verify_conv2d_conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                    padh, padw, strideh, stridew):
    _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                    padh, padw, strideh, stridew)
  def verify_conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads,
                    dilations):
    conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads,
                    dilations)

  """
  The UT for cce cross
  """
  print("============================================================")
  # ini.set_ddk_version()
  print("[ UNITTEST START test_conv2d ]")


  try:
      verify_conv2d_calc_para_from_dict({"ori_shape": (71,1,7,1), "dtype": "float16", "ori_format": "NCHW"},
                                      {"ori_shape": (3,71,34,172), "dtype": "float16", "ori_format": "NCHW"},
                                      (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1),
                                      {"ori_shape": (3,71,34,172), "dtype": "float16", "ori_format": "NCHW"})
  except RuntimeError:
      print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
      pass

  try:
      verify_conv2d_calc_para_from_dict({"ori_shape": (71,1,7,1), "dtype": "float16", "ori_format": "NCHW"},
                                      {"ori_shape": (3,71,34,172), "dtype": "float16", "ori_format": "NCHW"},
                                      (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1),
                                      {"ori_shape": (3,71,34,172), "dtype": "float16", "ori_format": "NCHW"})
  except RuntimeError:
      print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
      pass

  try:
    verify_conv2d_calc_para_from_dict({"ori_shape": (71,1,7), "dtype": "float16", "ori_format": "NCHW"},
                                    {"ori_shape": (3,71,34,172), "dtype": "float16", "ori_format": "NCHW"},
                                    (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1),
                                    {"ori_shape": (3,71,34,172), "dtype": "float16", "ori_format": "NCHW"})

  except RuntimeError:
      print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
      pass

  try:
    verify_conv2d_calc_para_from_dict({"ori_shape": (71,1,1, 1), "dtype": "float16", "ori_format": "NCHW"},
                                    {"ori_shape": (3,3,1,1), "dtype": "float16", "ori_format": "NCHW", "format":"FRACTAL_Z_C04"},
                                    (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1),
                                    {"ori_shape": (3,71,34,172), "dtype": "float16", "ori_format": "NCHW"})
  except RuntimeError:
      print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
      pass

  try:
    op_select_format({"ori_shape": (71,1,1, 1), "dtype": "float16", "ori_format": "NC"},
    {"ori_shape": (3,3,1,1), "dtype": "float16", "ori_format": "NCHW", "format":"FRACTAL_Z_C04"},
    None, (0, 0, 0, 0), None, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1))
  except RuntimeError:
      print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
      pass

  # input length invalid
  try:
    _conv_layer_cce(shape_in = (1, 16, 8, 8, 9), shape_w = (16, 16, 1, 1),
      in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
                padh = 0, padw = 0, strideh = 1, stridew = 1)
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass

  # stride length invalid
  try:
    _conv_layer_cce(shape_in = (1, 16, 8, 8), shape_w = (16, 16, 1, 1),
      in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
                padh = 0, padw = 0, strideh = (1, 1, 1), stridew = 1)
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass

  # weight length invalid
  try:
    _conv_layer_cce(shape_in = (1, 16, 8, 8), shape_w = (16, 16, 1, 1, 9),
      in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
                padh = 0, padw = 0, strideh = 1, stridew = 1)
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass
  # in_dtype invalid
  try:
    _conv_layer_cce(shape_in = (1, 16, 8, 8), shape_w = (16, 16, 1, 1, 9),
      in_dtype = "float165", w_dtype = "float16", res_dtype = "float16",
                padh = 0, padw = 0, strideh = 1, stridew = 1)
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass

  # filter_h filter_w invalid when weight unzip
  try:
    cceconf.cce_conf.te_set_version("Hi3796CV300ES")
    _conv_layer_compress_cce(shape_in = (1, 11456, 3418, 38), \
      shape_w = (512, 11456, 61, 20), \
      shape_index = (512, 11456, 61, 20), \
      in_dtype = "int8", w_dtype = "int8", index_dtype = "int8", \
      res_dtype = "int32", padh = 0, padw = 0, strideh = 1, stridew = 1)
    cceconf.cce_conf.te_set_version("Ascend310")
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass

  # filter_h filter_w invalid when weight unzip
  try:
    cceconf.cce_conf.te_set_version("Hi3796CV300ES")
    dilations = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    strides = [0, 0, 1, 1]
    shape_in = (1, 358, 3418, 38, 32)
    shape_w = (436760, 32, 16, 32)
    orig_shape_w = (512, 11456, 61, 20)
    q_offset = 1
    fm = tvm.placeholder(shape_in, name='fm', dtype='int8', attrs={'ori_format': 'NCHW'})
    filter_w = tvm.placeholder(shape_w, name='Filter', dtype='int8',
                                attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
    bias_tensor = None
    compress_index_shape = tvm.var("compress_index_shape", dtype="int32")
    compress_index = tvm.placeholder((compress_index_shape,),
                                      name='compress_index', dtype="int8")
    conv_res = conv2dcompress_compute(fm, filter_w, compress_index, \
        bias_tensor, None, None, strides, pads, dilations, 1, 'NCHW', \
        q_offset)
    cceconf.cce_conf.te_set_version("Ascend310")
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass

  # input format invalid

  try:
    conv2d_compute(tvm.placeholder((1, 1, 8, 8, 16), name = "fm", dtype = "float16", attrs = {"ori_shape":(1, 16, 8, 8), "ori_format":"NHW"}),
      tvm.placeholder((1, 1, 16, 16), name = "weight", dtype = "float16", attrs = {"ori_shape":(16, 16, 1, 1), "ori_format":"NCHW"}),
      bias = None, offset_w = (0, 0, 0, 0), outputs = None, strides = (1, 1, 1, 1), pads = (0, 0, 0, 0), dilations = (1, 1, 1, 1))
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass
  # input length invalid

  try:
    conv2d_compute(tvm.placeholder((1, 1, 8, 8, 16, 9), name = "fm", dtype = "float16", attrs = {"ori_shape":(1, 16, 8, 8), "ori_format":"NCHW"}),
      tvm.placeholder((1, 1, 16, 16), name = "weight", dtype = "float16", attrs = {"ori_shape":(16, 16, 1, 1), "ori_format":"NCHW"}),
      bias = None, offset_w = (0, 0, 0, 0), outputs = None, strides = (1, 1, 1, 1), pads = (0, 0, 0, 0), dilations = (1, 1, 1, 1))
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass
  # weight length invalid

  try:
    conv2d_compute(tvm.placeholder((1, 1, 8, 8, 16), name = "fm", dtype = "float16", attrs = {"ori_shape":(1, 16, 8, 8), "ori_format":"NCHW"}),
      tvm.placeholder((1, 1, 16, 16, 9), name = "weight", dtype = "float16", attrs = {"ori_shape":(16, 16, 1, 1), "ori_format":"NCHW"}),
      bias = None, offset_w = (0, 0, 0, 0), outputs = None, strides = (1, 1, 1, 1), pads = (0, 0, 0, 0), dilations = (1, 1, 1, 1))
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass
  # weight format invalid

  try:
    conv2d_compute(tvm.placeholder((1, 1, 8, 8, 16), name = "fm", dtype = "float16", attrs = {"ori_shape":(1, 16, 8, 8), "ori_format":"NCHW"}),
      tvm.placeholder((1, 1, 16, 16), name = "weight", dtype = "float16", attrs = {"ori_shape":(16, 16, 1, 1), "ori_format":"NCH"}),
      bias = None, offset_w = (0, 0, 0, 0), outputs = None, strides = (1, 1, 1, 1), pads = (0, 0, 0, 0), dilations = (1, 1, 1, 1))
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
    pass
  # stride length invalid

  try:
    conv2d_compute(tvm.placeholder((1, 1, 8, 8, 16), name = "fm", dtype = "float16", attrs = {"ori_shape":(1, 16, 8, 8), "ori_format":"NCHW"}),
      tvm.placeholder((1, 1, 16, 16), name = "weight", dtype = "float16", attrs = {"ori_shape":(16, 16, 1, 1), "ori_format":"NCHW"}),
      bias = None, offset_w = (0, 0, 0, 0), outputs = None, strides = (1, 1, 1, 1, 1), pads = (0, 0, 0, 0), dilations = (1, 1, 1, 1))
  except RuntimeError:
    print("[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))
  pass
  print("============================================================")
print("adding Conv2D testcase testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_conv_cce)