#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SubSampleLabels", "impl.sub_sample_labels", "sub_sample_labels")


def get_dict(_shape, dtype="int32"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": "ND", "ori_shape": _shape, "ori_format": "ND"}


def get_impl_list(labels_shape,
                  shuffle_matrix_shape,
                  output_shape,
                  batch_size_per_images,
                  positive_fraction):
    input_list = [get_dict(labels_shape),
                  get_dict(shuffle_matrix_shape)]

    output_list = [get_dict(output_shape)]

    par_list = [batch_size_per_images, positive_fraction]

    return input_list + output_list + par_list


case1 = {"params": get_impl_list((10,),
                                 (10,),
                                 (10,),
                                 6,
                                 0.5),
         "case_name": "sub_sample_labels_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": get_impl_list((41153,),
                                 (41153,),
                                 (41153,),
                                 256,
                                 0.5),
         "case_name": "sub_sample_labels_case_2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)
