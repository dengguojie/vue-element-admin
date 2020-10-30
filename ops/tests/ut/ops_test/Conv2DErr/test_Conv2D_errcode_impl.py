#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_dx_errorcode(test_arg):
    from te.utils.error_manager import error_manager_util as err_man
    import sys
    sys.path.append("./llt/ops/ut/testcase_python")

    succ_str = "OK"
    fail_str = "FAILED"

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e60108]")
    dict_args = {}
    dict_args['errCode'] = "E60108"
    dict_args['op_name'] = "A"
    dict_args['reason'] = "B"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e60114]")
    dict_args = dict()
    dict_args["errCode"] = "E60114"
    dict_args["op_name"] = "x"
    dict_args["reason"] = "NCHW"
    dict_args["value"] = "NC1HWC0"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61300]")
    dict_args = dict()
    dict_args["errCode"] = "E61300"
    dict_args["op_name"] = "conv2d"
    dict_args["param_name"] = "A"
    dict_args["expect_value"] = "1"
    dict_args["condition"] = "-1"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61301]")
    dict_args = {}
    dict_args['errCode'] = "E61301"
    dict_args["op_name"] = "x"
    dict_args["param_name_1"] = "A"
    dict_args["param_name_2"] = "B"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61203]")
    dict_args = {}
    dict_args['errCode'] = "E61203"
    dict_args["op_name"] = "x"
    dict_args["sence_params"] = "A"
    dict_args["param_1"] = "1"
    dict_args["param_2"] = "2"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61204]")
    dict_args = {}
    dict_args['errCode'] = "E61204"
    dict_args["op_name"] = "x"
    dict_args["sence_params"] = "A"
    dict_args["param_1"] = "1"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61600]")
    dict_args = {}
    dict_args['errCode'] = "E61600"
    dict_args["op_name"] = "x"
    dict_args["param_name"] = "A"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61601]")
    dict_args = {}
    dict_args['errCode'] = "E61601"
    dict_args["op_name"] = "x"
    dict_args["scene"] = "A"
    dict_args["param_name"] = "B"
    dict_args["claim"] = "2"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61602]")
    dict_args = {}
    dict_args['errCode'] = "E61602"
    dict_args["op_name"] = "x"
    dict_args["param_name"] = "A"
    dict_args["optype_1"] = "int8"
    dict_args["optype_2"] = "float"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d_backprop_filter_errCode e61603]")
    dict_args = {}
    dict_args['errCode'] = "E61603"
    dict_args["op_name"] = "x"
    dict_args["param_1"] = "A"
    dict_args["param_2"] = "int8"
    err_man.get_error_message(dict_args)
    print(err_man.get_error_message(dict_args))
    print("[ %s ].%s" % (succ_str, sys._getframe().f_code.co_name))
    print("")
    print("---------------------------------------------------")

print("adding Conv2D errcode testcases")
ut_case.add_cust_test_func(test_func=test_dx_errorcode)
