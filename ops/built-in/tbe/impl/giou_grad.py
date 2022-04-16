# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
giou_grad
"""

from te import tik
from te.utils import para_check
from impl.diou_grad import Constant
from impl.diou_grad import DIoUGrad


# 'pylint: disable=too-many-statements,too-many-arguments
class GIoUGrad(DIoUGrad):
    """GIoUGrad"""

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, dy, bboxes, trans, kernel_name):
        """__init__"""
        super().__init__(dy, bboxes, trans, kernel_name)

    def compute_core(self, task_idx):
        """giou_grad_compute_compute_core"""
        # func: init all unit
        self.init_date()
        self.enclose = self.tik_instance.Tensor("float32", [self.data_align], name="enclose", scope=tik.scope_ubuf)
        self.denclose = self.tik_instance.Tensor("float32", [self.data_align], name="denclose", scope=tik.scope_ubuf)
        # func: get b1 and b2
        self.move_in(task_idx)

        # func: compute for inter/union
        self.update_part()
        # func: compute for enclose
        self.tik_instance.vec_mul(Constant.BLOCK, self.enclose, self.cw, self.ch, self.rep_time, 1, 1, 1)

        # func: compute for dinter/dunion/denclose
        self.update_dpart()

        # func: compute for dbboxes/dgtboxes in inter
        self.inter_part()

        # func: compute for dbboxes/dgtboxes in union
        self.union_part()

        # func: compute for dbboxes/dgtboxes in enclose
        self.enclose_part()

        # func: resite res for attr_trans
        self.update_dboxes(task_idx)

    def update_dpart(self):
        """update_dpart"""
        # `for dunion, dunion = (1 / enclose - inter / (union ** 2)) * dy`
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_a, self.dy_ub, self.enclose,
                               self.rep_time, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_b, self.inter, self.union, self.rep_time,
                               1, 1, 1, 1, 1, 1)
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_c, self.tmp_b, self.union, self.rep_time,
                               1, 1, 1, 1, 1, 1)
        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_d, self.dy_ub, self.tmp_c, self.rep_time, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.dunion, self.tmp_a, self.tmp_d, self.rep_time, 1, 1, 1)

        # `for dinter, dinter = 1 / union * dy - dunion`
        self.tik_instance.vdiv(Constant.BLOCK, self.dinter, self.dy_ub, self.union, self.rep_time, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.dinter, self.dinter, self.dunion, self.rep_time, 1, 1, 1)

        # `for denclose, denclose = -(union / (enclose ** 2)) * dy`
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_a, self.union, self.enclose,
                               self.rep_time, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_b, self.tmp_a, self.enclose,
                               self.rep_time, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_c, self.dy_ub, self.tmp_b, self.rep_time, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.denclose, self.tmp_zero, self.tmp_c, self.rep_time, 1, 1, 1)

    def enclose_part(self):
        """enclose_part"""
        # for enclose part
        self.tik_instance.vec_mul(Constant.BLOCK, self.dxlen, self.denclose, self.ch, self.rep_time, 1, 1, 1)  # max_x
        self.tik_instance.vec_mul(Constant.BLOCK, self.dylen, self.denclose, self.cw, self.rep_time, 1, 1, 1)  # max_y

        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_a, self.tmp_zero, self.dxlen, self.rep_time, 1, 1,
                                  1)  # min_x
        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_b, self.tmp_zero, self.dylen, self.rep_time, 1, 1,
                                  1)  # min_y

        with self.tik_instance.for_range(0, self.data_align // Constant.MASK_BLOCK) as idx:
            # `for enclose part : max(b1_x2, b2_x2)`
            self.tik_instance.vec_cmpv_gt(self.mask, self.b1x2[Constant.MASK_BLOCK * idx],
                                          self.b2x2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c, self.mask,
                                      self.dxlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1x2[Constant.MASK_BLOCK * idx],
                                      self.tmp_c, self.db1x2[Constant.MASK_BLOCK * idx], 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d, self.mask,
                                      self.tmp_zero, self.dxlen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2x2[Constant.MASK_BLOCK * idx],
                                      self.tmp_d, self.db2x2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            # `for enclose part : min(b1_x1, b2_x1)`
            self.tik_instance.vec_cmpv_lt(self.mask, self.b1x1[Constant.MASK_BLOCK * idx],
                                          self.b2x1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                          Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c, self.mask,
                                      self.tmp_a[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1x1[Constant.MASK_BLOCK * idx],
                                      self.tmp_c, self.db1x1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d, self.mask,
                                      self.tmp_zero, self.tmp_a[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2x1[Constant.MASK_BLOCK * idx],
                                      self.tmp_d, self.db2x1[Constant.MASK_BLOCK * idx], 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            # `for enclose part : max(b1_y2, b2_y2)`
            self.tik_instance.vec_cmpv_gt(self.mask, self.b1y2[Constant.MASK_BLOCK * idx],
                                          self.b2y2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c, self.mask,
                                      self.dylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1y2[Constant.MASK_BLOCK * idx],
                                      self.tmp_c, self.db1y2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d, self.mask,
                                      self.tmp_zero, self.dylen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2y2[Constant.MASK_BLOCK * idx],
                                      self.tmp_d, self.db2y2[Constant.MASK_BLOCK * idx],
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)

            # `for enclose part : min(b1_y1, b2_y1)`
            self.tik_instance.vec_cmpv_lt(self.mask, self.b1y1[Constant.MASK_BLOCK * idx],
                                          self.b2y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                          Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c, self.mask,
                                      self.tmp_b[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1y1[Constant.MASK_BLOCK * idx],
                                      self.tmp_c, self.db1y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d, self.mask,
                                      self.tmp_zero, self.tmp_b[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2y1[Constant.MASK_BLOCK * idx],
                                      self.tmp_d, self.db2y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def giou_grad(dy, bboxes, gtboxes, dbboxes, dgtboxes, trans=False, is_cross=True, mode="iou",
              kernel_name="giou_grad"):
    """
    calculating data

    Parameters
    ----------
    Inputs:
    dy : dict
        data of grad increment, shape must be [n].
        source data type, support "float32"
    bboxes : dict
        shape and dtype of bboxes, the coordinates of bbox
        shape must be [4, n]
        [x1, y1, x2, y2] or [x, y, w, h]
    gtboxes : dict
        shape and dtype of gtboxes, the coordinates of gtbox
        shape must be [4, m]
        [x1, y1, x2, y2] or [x, y, w, h]

    Outputs:
    dbboxes : dict
        shape and dtype of dbboxes, the coordinates of dbbox
        shape must be [4, n]
        [x1, y1, x2, y2] or [x, y, w, h]
    dgtboxes : dict
        shape and dtype of dgtboxes, the coordinates of dgtbox
        shape must be [4, m]
        [x1, y1, x2, y2] or [x, y, w, h]

    Attributes:
    trans : bool
        true for 'xywh', false for 'xyxy'
    is_cross : bool
        if false: m must be equal to n
    mode :  str
        ('iou','iof')
        iou : the output is inter_area / total_area
        iof : the output is inter_area / gtboxes_area
    kernel_name : str
        kernel name, default value is "giou_grad"
    Returns
    -------
    None
    """
    op_obj = GIoUGrad(dy, bboxes, trans, kernel_name)

    return op_obj.compute()
