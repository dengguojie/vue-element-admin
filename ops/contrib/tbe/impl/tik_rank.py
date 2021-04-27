# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""
from te import tik

BLOCK_NUM = 8  # block number proceed by each ai core
UB_SIZE = 240 * 1024  # size of 310 ai core ub buffer
AI_COER_NUM = 2


class Rank():
    """
    Parameters
    ----------
    kernel_name : kernel name, default value is "Rank"
    function_description : return the order of input
    input : dict shape dtype format origin_format of input
    output : dict shape dtype format origin_format of output

    Returns
    -------
    None

    """

    def __init__(self, input0, output0, kernel_name="Rank"):
        self.shape = input0.get("shape")
        self.dtype = input0.get("dtype")
        self.rank = len(self.shape)

        if self.rank > 5:
            raise RuntimeError("order of above five is invalid ")

        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.block_bite_size = 32
        self.ub_tensor_size = self.block_bite_size // 4

        self.input_gm = self.tik_instance.Tensor(self.dtype, self.shape,
                                                 name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor('int32', (self.ub_tensor_size,),
                                                  name="output_gm", scope=tik.scope_gm)

    def tilling_mode_select(self):
        self.mode = 1

    def rank_compute(self):
        self.input_ub = self.tik_instance.Tensor(self.dtype, (16,),
                                                 name="input_ub", scope=tik.scope_ubuf)

        self.tik_instance.data_move(self.input_ub, self.input_gm, 0, 1, 1, 0, 0)
        self.output_ub = self.tik_instance.Tensor('int32', (8,),
                                                  name="output_ub", scope=tik.scope_ubuf)

        self.rank_sclr = self.tik_instance.Scalar(init_value=self.rank, dtype="int32")
        self.tik_instance.vector_dup(8, self.output_ub, self.rank_sclr, 1, 1, 8)
        self.tik_instance.data_move(self.output_gm, self.output_ub, 0, 1, 1, 0, 0)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm],
                                   outputs=[self.output_gm], enable_l2=True)
        return self.tik_instance


def tik_rank(input0, output0, kernel_name="Rank"):
    obj = Rank(input0, output0, kernel_name)
    obj.rank_compute()
