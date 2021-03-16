#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("EmbeddingBag", "impl.embedding_bag", "embedding_bag")


def get_dict(_shape, dtype="float32"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": "ND", "ori_shape": _shape, "ori_format": "ND"}


def get_impl_list(weight_shape, indices_shape,
                  offset_shape, per_sample_shape,
                  output_shape, mode,
                  scale_grid_by_freq, sparse,
                  include_last_offset):
    input_list = [get_dict(weight_shape),
                  get_dict(indices_shape, "int32"),
                  get_dict(offset_shape, "int32"),
                  get_dict(per_sample_shape)]

    output_list = [get_dict(output_shape)]

    par_list = [mode, scale_grid_by_freq, sparse, include_last_offset]
    return input_list + output_list + par_list


case1 = {"params": get_impl_list((100, 3),
                                 (10,),
                                 (3,),
                                 (10,),
                                 (3, 3),
                                 "sum",
                                 False,
                                 False,
                                 False),
         "case_name": "embedding_bag_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": get_impl_list((100, 3),
                                 (10,),
                                 (3,),
                                 (10,),
                                 (3, 3),
                                 "mean",
                                 False,
                                 False,
                                 False),
         "case_name": "embedding_bag_case_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": get_impl_list((100, 3),
                                 (10,),
                                 (3,),
                                 (10,),
                                 (3, 3),
                                 "max",
                                 False,
                                 False,
                                 False),
         "case_name": "embedding_bag_case_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": get_impl_list((100, 3),
                                 (3, 5),
                                 (3,),
                                 (10,),
                                 (3, 3),
                                 "sum",
                                 False,
                                 False,
                                 False),
         "case_name": "embedding_bag_case_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend310"], case3)
ut_case.add_case(["Ascend310"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)
