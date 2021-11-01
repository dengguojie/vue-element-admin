"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

UpdateTensorDesc ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from tbe.common.platform import set_current_compile_soc_info
from impl.update_tensor_desc import update_tensor_desc

ut_case = OpUT("UpdateTensorDesc", "impl.update_tensor_desc", "update_tensor_desc")

def test_1951_update_tensor_desc_001(test_arg):
    set_current_compile_soc_info('Ascend710')
    update_tensor_desc({'shape': (128, ), 'dtype': 'int64', 'format': 'ND', 'ori_shape': (128, ), 'ori_format': 'ND'},
                       [2, 4, 6, 16],
                       "test_1951_update_tensor_desc_001")
    set_current_compile_soc_info(test_arg)

ut_case.add_cust_test_func(test_func=test_1951_update_tensor_desc_001)


if __name__ == '__main__':
    ut_case.run("Ascend710")
