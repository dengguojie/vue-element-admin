import os
import sys

cur_dir = os.path.dirname(__file__)
llt_dir = os.path.dirname(cur_dir)

if llt_dir not in sys.path:
    sys.path.append(llt_dir)

from tools.auto_gen_op_cases import auto_gen_broadcast_op_ut, auto_gen_reduce_op_ut, auto_gen_single_elewise_op_ut, auto_gen_grad_broadcast_op_ut
from common.config.llt_config import LLTConf


def auto_gen_broadcast_op_ut_by_ini(op_ini_path):
    broadcast_op_list = {}
    grad_broadcast_op_list = {}
    format_agnostic_op_list = {}
    reduce_op_list = {}
    ini_path = op_ini_path

    with open(ini_path) as ini_file:
        lines = ini_file.readlines()

    op_type=None
    input_idx_list = []
    dtypes = None
    op_pattern = None
    has_attr = False
    op_file = None
    op_intf = None
    for line in lines:
        line = line.strip()
        if line.startswith("["):
            if op_pattern == "broadcast" and len(input_idx_list) == 2 and dtypes is not None and not has_attr:
                dtype_list = []
                for dt in dtypes.split(","):
                    if dt == "float":
                        dtype_list.append("float32")
                    else:
                        dtype_list.append(dt)
                if "Grad" not in op_type:
                    broadcast_op_list[op_type] = [[{
                        "soc" : ["Ascend910", ],
                        "dtype" : dtype_list
                    },], op_file, op_intf]
                else:
                    grad_broadcast_op_list[op_type] = [[{
                        "soc" : ["Ascend910", ],
                        "dtype" : dtype_list
                    },], op_file, op_intf]
            elif op_pattern == "formatAgnostic" and len(input_idx_list)==1 and dtypes is not None and not has_attr:
                dtype_list = []
                for dt in dtypes.split(","):
                    if dt == "float":
                        dtype_list.append("float32")
                    else:
                        dtype_list.append(dt)
                format_agnostic_op_list[op_type] = [[{
                    "soc" : ["Ascend910", ],
                    "dtype" : dtype_list
                },], op_file, op_intf]
            elif op_pattern == "reduce" and len(input_idx_list) == 1 and dtypes is not None:
                dtype_list = []
                for dt in dtypes.split(","):
                    if dt == "float":
                        dtype_list.append("float32")
                    else:
                        dtype_list.append(dt)
                reduce_op_list[op_type] = [[{
                    "soc" : ["Ascend910", ],
                    "dtype" : dtype_list
                },], op_file, op_intf]

            op_type = line[1:-1]
            op_pattern = None
            input_cnt = 0
            input_idx_list = []
            dtypes = None
            has_attr = False
            op_file = None
            op_intf = None
        elif line.startswith("input"):
            input_idx = line[:line.index(".")]
            if input_idx not in input_idx_list:
                input_idx_list.append(input_idx)
            if "dtype" in line:
                tmp_dtypes = line[line.index("dtype")+6:]
                if dtypes is None or dtypes == tmp_dtypes:
                    dtypes = tmp_dtypes
                else:
                    dtypes = None
        elif line.startswith("output"):
            if "dtype" in line:
                tmp_dtypes = line[line.index("dtype")+6:]
                if dtypes is None or dtypes == tmp_dtypes:
                    dtypes = tmp_dtypes
                else:
                    dtypes = None
        elif line.startswith("attr"):
            has_attr = True
        elif line.startswith("opFile"):
            op_file = line[13:]
        elif line.startswith("opInterface"):
            op_intf = line[18:]
        elif line.startswith("op.pattern"):
            op_pattern = line[11:]
        else:
            continue

    for op in broadcast_op_list.keys():
        auto_gen_broadcast_op_ut(op, broadcast_op_list[op][0], broadcast_op_list[op][1], broadcast_op_list[op][2])

    for op in grad_broadcast_op_list.keys():
        auto_gen_grad_broadcast_op_ut(op, grad_broadcast_op_list[op][0], grad_broadcast_op_list[op][1], grad_broadcast_op_list[op][2])

    for op in format_agnostic_op_list.keys():
        auto_gen_single_elewise_op_ut(op, format_agnostic_op_list[op][0], format_agnostic_op_list[op][1], format_agnostic_op_list[op][2])

    for op in reduce_op_list.keys():
        auto_gen_reduce_op_ut(op, reduce_op_list[op][0], reduce_op_list[op][1], reduce_op_list[op][2])


def auto_gen_ut_structure(op_type, frameworks="tf"):
    op_ut_root_dir = os.path.join(LLTConf.llt_root_path, "ut/ops_test")
    op_dir = os.path.join(op_ut_root_dir, op_type)

    def gen_proto_test_structure():
        proto_test_file = os.path.join(op_dir, "test_%s_proto.cpp" % op_type.lower())
        if os.path.exists(proto_test_file):
            print("already exist proto test file: %s, don't create it" % proto_test_file)
            return

        proto_str="""#include <gtest/gtest.h>
#include <iostream>
#include "../../../../../../ops/built-in/op_proto/inc/all_ops.h"
#include "../../../common/ge_util/op_test_util.h"

class %sProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "%s Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "%s Proto Test TearDown" << std::endl;
  }
};

class ge::op::%s;

TEST_F(%sProtoTest, xxxx){
// TODO add you test code
}

""" % (op_type, op_type, op_type, op_type, op_type)
        with open(proto_test_file, "w+") as p_f:
            p_f.write(proto_str)

    def gen_plugin_test_structure(fwk_type):
        plugin_test_file = os.path.join(op_dir, "test_%s_%s_plugin.cpp" % (op_type.lower(), fwk_type.lower()))
        plugin_test_proto_dir = os.path.join(op_dir, "ops_test_proto_%s" % fwk_type)
        if os.path.exists(plugin_test_file):
            print("already exist %s plugin test file: %s, don't create it" % (op_type, plugin_test_file))
            return
        if os.path.exists(plugin_test_proto_dir):
            print("already exist plugin test proto file dir, don't create it")
        else:
            os.makedirs(plugin_test_proto_dir)
        plugin_str = """"""
        with open(plugin_test_file, "w+") as p_f:
            p_f.write(plugin_str)

    def gen_impl_test_structure():
        impl_test_file = os.path.join(op_dir, "test_%s_impl.py" % op_type.lower())
        if os.path.exists(impl_test_file):
            print("already exist %s impl test file: %s, don't create it" % (op_type, impl_test_file))
            return

        impl_str="""#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("%s", None, None)

ut_case.add_test_cfg_cov_case("all")
# TODO add you test case

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
""" % op_type
        with open(impl_test_file, "w+") as i_f:
            i_f.write(impl_str)

    if not os.path.exists(op_dir):
        os.makedirs(op_dir)
    frameworks = frameworks.split(",")
    gen_proto_test_structure()
    gen_impl_test_structure()
    if "tf" in frameworks:
        gen_plugin_test_structure("tf")
    if "caffe" in frameworks:
        gen_plugin_test_structure("caffe")


if __name__ == "__main__":
    # auto_gen_ut_structure("InplaceAdd", "tf")
    # auto_gen_ut_structure("MaxPool", "tf")
    # auto_gen_broadcast_op_ut_by_ini()
    params = sys.argv[1:]
    if len(params) == 2:
        auto_gen_ut_structure(params[0], params[1])
    else:
        print("python exec arg len is not 2, should contains op type and framework type")