# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from . import get_version

tik, TBE_VERSION = get_version.get_tbe_version()


def ceil_div_offline(value, factor):
    return (value + factor - 1) // factor


class ScopeMap1:
    """
    Parameters
    ----------
    kernel_name : kernel name, default value is "map1"
    function_description : return the absolute value of normalized data
                           only supports input shape (1, 100, 4) or (1, 300 ,4)
    input0 : dict shape dtype format of normalized data
    output0 : dict shape dtype format of absolute data
    Returns
    -------
    None
    """

    def __init__(self, input0, output0, kernel_name="map1"):
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.max_box_num = input0.get('shape')[1]
        self.coor_num = input0.get('shape')[2]
        proposal_norm_shape = input0.get('shape')
        proposal_abso_shape = output0.get('shape')

        if proposal_norm_shape != (1, 100, 4) and proposal_norm_shape != (1, 300, 4):
            raise RuntimeError("input0 shape not valid")
        if input0.get('dtype') != 'float16' or output0.get('dtype') != 'float16':
            raise RuntimeError("data type should be float16")
        if proposal_abso_shape != proposal_norm_shape:
            raise RuntimeError("output0 shape should be the same as input0's")

        if self.max_box_num == 100:
            self.image_h = 600.
        else:
            self.image_h = 320.
        self.image_w = 1024.

        self.pad_max_box_num = ceil_div_offline(self.max_box_num, 16) * 16
        self.proposal_norm_in_gm = self.tik_instance.Tensor("float16", proposal_norm_shape,
                                                            name="proposal_norm_in_gm",
                                                            scope=tik.scope_gm)
        self.proposal_abso_out_gm = self.tik_instance.Tensor("float16", proposal_abso_shape,
                                                             name="proposal_abso_out_gm",
                                                             scope=tik.scope_gm)

    def tilling_mode_select(self):
        self.mode = 1

    def global_init(self):
        pass

    def map1_compute(self):
        proposal_norm_in_ub = self.tik_instance.Tensor("float16", (
            1, self.pad_max_box_num, ceil_div_offline(self.coor_num, 16) * 16),
                                                       name="proposal_norm_in_ub",
                                                       scope=tik.scope_ubuf)

        coor = self.tik_instance.Tensor("float16", (
            ceil_div_offline(self.coor_num, 16) * 16, self.pad_max_box_num), name="coor",
                                        scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.max_box_num) as mov_t:
            self.tik_instance.data_move(proposal_norm_in_ub[0, mov_t, 0],
                                        self.proposal_norm_in_gm[0, mov_t, 0], 0, 1, 1, 0, 0)

        dst_list = [coor[i, 0] for i in range(0, 16)]
        src_list = [proposal_norm_in_ub[0, i, 0] for i in range(0, 16)]
        self.tik_instance.vnchwconv(True, True, dst_list, src_list, self.pad_max_box_num // 16, 1,
                                    16)

        repeat_time = self.max_box_num // 128
        leftmask = self.max_box_num % 128
        if repeat_time > 0:
            self.tik_instance.vmuls(128, coor[0, 0], coor[0, 0], self.image_h, repeat_time, 1, 1, 8,
                                    8)
            self.tik_instance.vmuls(128, coor[1, 0], coor[1, 0], self.image_w, repeat_time, 1, 1, 8,
                                    8)
            self.tik_instance.vmuls(128, coor[2, 0], coor[2, 0], self.image_h, repeat_time, 1, 1, 8,
                                    8)
            self.tik_instance.vmuls(128, coor[3, 0], coor[3, 0], self.image_w, repeat_time, 1, 1, 8,
                                    8)
        if leftmask > 0:
            self.tik_instance.vmuls(leftmask, coor[0, self.max_box_num - leftmask],
                                    coor[0, self.max_box_num - leftmask], self.image_h, 1, 1, 1, 8,
                                    8)
            self.tik_instance.vmuls(leftmask, coor[1, self.max_box_num - leftmask],
                                    coor[1, self.max_box_num - leftmask], self.image_w, 1, 1, 1, 8,
                                    8)
            self.tik_instance.vmuls(leftmask, coor[2, self.max_box_num - leftmask],
                                    coor[2, self.max_box_num - leftmask], self.image_h, 1, 1, 1, 8,
                                    8)
            self.tik_instance.vmuls(leftmask, coor[3, self.max_box_num - leftmask],
                                    coor[3, self.max_box_num - leftmask], self.image_w, 1, 1, 1, 8,
                                    8)

        dst_list = [proposal_norm_in_ub[0, i, 0] for i in range(0, 16)]
        src_list = [coor[i, 0] for i in range(0, 16)]
        self.tik_instance.vnchwconv(True, True, dst_list, src_list, self.pad_max_box_num // 16, 16,
                                    1)

        with self.tik_instance.for_range(0, self.max_box_num) as mov_t:
            self.tik_instance.data_move(self.proposal_abso_out_gm[0, mov_t, 0],
                                        proposal_norm_in_ub[0, mov_t, 0], 0, 1, 1, 0, 0)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.proposal_norm_in_gm],
                                   outputs=[self.proposal_abso_out_gm], enable_l2=False)
        return self.tik_instance


def fasterrcnn_map1_tik(input0, output0, kernel_name="map1"):
    """
    Calculate the absolute coordinates of the input box.
    """
    obj = ScopeMap1(input0, output0, kernel_name)
    obj.map1_compute()
