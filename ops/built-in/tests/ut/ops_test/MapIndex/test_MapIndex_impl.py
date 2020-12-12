#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("MapIndexz", "impl.map_index", "map_index")


def gen_static_aipp_case(x_shape, data_seq_shape, y_shape,
                         dtype, format, case_name_val, expect):
    return {"params": [{"shape": x_shape, "dtype": dtype, "ori_shape": x_shape, "ori_format": format, "format": format},
                       {"shape": data_seq_shape, "dtype": dtype, "ori_shape": data_seq_shape, "ori_format": format, "format": format},
                       None,
                       {"shape": y_shape, "dtype": dtype, "ori_shape": y_shape, "ori_format": format, "format": format}],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_static_aipp_case([1], [100], [8],
                                      "int32", "ND", "mapindex_1", "success"))

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run("Ascend310")
    exit(0)
