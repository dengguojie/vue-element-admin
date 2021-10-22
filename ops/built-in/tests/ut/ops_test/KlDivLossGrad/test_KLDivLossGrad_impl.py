"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

KLDiv ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("KlDivLossGrad", "impl.kl_div_loss_grad", "kl_div_loss_grad")

# ut_case.add_test_cfg_cov_case("all")
ut_case.add_case("all", {
    "params": [{'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (32, 32), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (32, 32), 'ori_format': 'ND'},
               {'shape': (32, 32), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (32, 32), 'ori_format': 'ND'},
               {'shape': (32, 32), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (32, 32), 'ori_format': 'ND'},
               "mean",
               False],
    "expect": "success"
})

if __name__ == "__main__":
    ut_case.run("Ascend910A")


