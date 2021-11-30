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
"""
giou_grad
"""

from te.platform.fusion_manager import fusion_manager
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    BLOCK = 8
    MASK_BLOCK = 64
    BOX_LOC = 4
    MINI_BATCH = 512
    REP_TIME = 64
    REP_STRIDE = 8


# 'pylint: disable=too-many-statements,too-many-arguments
@fusion_manager.register("giou_grad")
class GIoUGrad:
    """GIoUGrad"""

    # 'pylint: disable=too-many-statements,too-many-arguments
    def __init__(self, dy, bboxes, gtboxes, trans, is_cross, mode, kernel_name):
        """__init__"""
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.all_num, self.dtype = self.paras_check(dy, bboxes, gtboxes, trans, is_cross, mode)

        self.task_num = (self.all_num + Constant.MINI_BATCH - 1) // Constant.MINI_BATCH

        self.all_num_align = self.task_num * Constant.MINI_BATCH
        self.move_rep = self.all_num_align // Constant.BLOCK

        self.move_flag = True
        if self.all_num_align == self.all_num:
            self.move_flag = False

        # func: apply for the input/output tensors
        self.dy = self.tik_instance.Tensor(self.dtype, [self.all_num], name="dy", scope=tik.scope_gm)
        self.bboxes = self.tik_instance.Tensor(self.dtype,
                                               [Constant.BOX_LOC,
                                                self.all_num],
                                               name="bboxes",
                                               scope=tik.scope_gm)
        self.gtboxes = self.tik_instance.Tensor(self.dtype,
                                                [Constant.BOX_LOC,
                                                 self.all_num],
                                                name="gtboxes",
                                                scope=tik.scope_gm)

        self.dbboxes = self.tik_instance.Tensor(self.dtype,
                                                [Constant.BOX_LOC,
                                                 self.all_num],
                                                name="dbboxes",
                                                scope=tik.scope_gm)
        self.dgtboxes = self.tik_instance.Tensor(self.dtype, [Constant.BOX_LOC, self.all_num], name="dgtboxes",
                                                 scope=tik.scope_gm)
        self.dbboxes_ = self.tik_instance.Tensor(self.dtype, [Constant.BOX_LOC, self.all_num_align], name="dbboxes_",
                                                 scope=tik.scope_gm, is_workspace=True)
        self.dgtboxes_ = self.tik_instance.Tensor(self.dtype, [Constant.BOX_LOC, self.all_num_align], name="dgtboxes_",
                                                  scope=tik.scope_gm, is_workspace=True)

        # func: apply for the calculation cache of inter/union/enclose
        self.inter = None
        self.union = None
        self.enclose = None

        # func: apply for the calculation cache of mask: record the result of compare api
        self.mask = None

        # func: apply for the calculation cache of b1x/b1y/b1w/b1h in bboxes
        self.b1x = None
        self.b1y = None
        self.b1w = None
        self.b1h = None

        # func: apply for the calculation cache of b2x/b2y/b2w/b2h in gtboxes
        self.b2x = None
        self.b2y = None
        self.b2w = None
        self.b2h = None

        # func: apply for the calculation cache of b1x1/b1x2/b1y1/b1y2/b2x1/b2x2/b2y1/b2y2
        self.b1x1 = None
        self.b1x2 = None
        self.b1y1 = None
        self.b1y2 = None
        self.b2x1 = None
        self.b2x2 = None
        self.b2y1 = None
        self.b2y2 = None

        # func: apply for the calculation cache of db1x/db1y/db1w/db1h in dbboxes
        self.db1x = None
        self.db1y = None
        self.db1w = None
        self.db1h = None

        # func: apply for the calculation cache of db2x/db2y/db2w/db2h in dgtboxes
        self.db2x = None
        self.db2y = None
        self.db2w = None
        self.db2h = None

        # func: apply for the calculation cache of db1x1/db1x2/db1y1/db1y2/db2x1/db2x2/db2y1/db2y2
        self.db1x1 = None
        self.db1x2 = None
        self.db1y1 = None
        self.db1y2 = None
        self.db2x1 = None
        self.db2x2 = None
        self.db2y1 = None
        self.db2y2 = None

        # func: apply for the calculation cache of zero
        self.tmp_zero = None

        # func: apply for the calculation cache of temp obj
        self.tmp_a = None
        self.tmp_b = None
        self.tmp_c = None
        self.tmp_d = None

        # func: apply for the calculation cache of dxlen/dylen
        self.dxlen = None
        self.dylen = None

        # func: apply for the calculation cache of dinter/dunion/denclose
        self.dinter = None
        self.dunion = None
        self.denclose = None

        # func: apply for the calculation cache of xlen_min/ylen_min/xlen_max/ylen_max
        self.xlen_min = None
        self.ylen_min = None
        self.xlen_max = None
        self.ylen_max = None

        # func: apply for the scalar obj of 0.5
        self.half = None

        # func: apply for the calculation cache of dy_ub/bboxes_ub/gtboxes_ub/dbboxes_ub/dgtboxes_ub
        self.dy_ub = None
        self.bboxes_ub = None
        self.gtboxes_ub = None
        self.dbboxes_ub = None
        self.dgtboxes_ub = None

        # func: for task allocation
        self.available_aicore_num = tik.Dprofile().get_aicore_num()
        self.used_aicore_num = \
            self.available_aicore_num if self.task_num > self.available_aicore_num else self.task_num
        self.batch_num_per_aicore = self.task_num // self.used_aicore_num
        self.batch_tail = self.task_num % self.used_aicore_num

    # 'pylint: disable=too-many-arguments
    def paras_check(self, dy, bboxes, gtboxes, trans, is_cross, mode):
        """paras_check"""
        shape_dy = dy.get("shape")
        dtype_dy = dy.get("dtype").lower()
        para_check.check_shape_rule(shape_dy)
        para_check.check_dtype_rule(dtype_dy, ("float32"))

        shape_bboxes = bboxes.get("shape")
        dtype_bboxes = bboxes.get("dtype").lower()
        para_check.check_shape_rule(shape_bboxes)
        para_check.check_dtype_rule(dtype_bboxes, ("float32"))

        shape_gtboxes = gtboxes.get("shape")

        if shape_bboxes != shape_gtboxes:
            raise RuntimeError("shape_bboxes should equal to shape_gtboxes.")

        para_check.check_kernel_name(self.kernel_name)

        if not trans:
            raise RuntimeError("The attr_trans should be true.")

        if is_cross:
            raise RuntimeError("The attr_is_cross should be false.")

        if mode != "iou":
            raise RuntimeError("The attr_mode should be 'iou'.")

        if shape_bboxes[0] != Constant.BOX_LOC:
            raise RuntimeError("The shape of bboxes should be [4, -1].")

        if shape_bboxes[1] != shape_dy[0]:
            raise RuntimeError("The value of bboxes_shape[1] should equal to dy_shape[0].")

        if dtype_dy != dtype_bboxes:
            raise RuntimeError("The dtype of bboxes should equal to dy.")

        return shape_dy[0], dtype_dy

    def compute(self):
        """giou_grad_compute"""
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)

        if self.move_flag:
            self.move_out()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.dy, self.bboxes, self.gtboxes],
                                   outputs=[self.dbboxes, self.dgtboxes])

        return self.tik_instance

    def compute_core(self, task_idx):
        """giou_grad_compute_compute_core"""
        # func: init all unit
        self.init_date()

        # func: get b1 and b2
        self.move_in(task_idx)

        # func: xyhw trans to xyxy
        self.data_trans()

        # func: compute for inter/union/enclose, `giou = inter/union + union/enclose - 1`
        self.update_part()

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

    # 'pylint: disable=too-many-statements
    def init_date(self):
        """init_date"""
        # func: create for the calculation cache of inter/union/enclose
        self.inter = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="inter", scope=tik.scope_ubuf)
        self.union = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="union", scope=tik.scope_ubuf)
        self.enclose = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="enclose",
                                                scope=tik.scope_ubuf)

        # func: create for the calculation cache of mask: record the result of compare api
        self.mask = self.tik_instance.Tensor("uint16", [Constant.BLOCK], name="mask", scope=tik.scope_ubuf)

        # func: create for the calculation cache of b1x/b1y/b1w/b1h in bboxes
        self.b1x = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1x", scope=tik.scope_ubuf)
        self.b1y = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1y", scope=tik.scope_ubuf)
        self.b1w = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1w", scope=tik.scope_ubuf)
        self.b1h = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1h", scope=tik.scope_ubuf)

        # func: create for the calculation cache of b2x/b2y/b2w/b2h in gtboxes
        self.b2x = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2x", scope=tik.scope_ubuf)
        self.b2y = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2y", scope=tik.scope_ubuf)
        self.b2w = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2w", scope=tik.scope_ubuf)
        self.b2h = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2h", scope=tik.scope_ubuf)

        # func: create for the calculation cache of b1x1/b1x2/b1y1/b1y2/b2x1/b2x2/b2y1/b2y2
        self.b1x1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1x1", scope=tik.scope_ubuf)
        self.b1x2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1x2", scope=tik.scope_ubuf)
        self.b1y1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1y1", scope=tik.scope_ubuf)
        self.b1y2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1y2", scope=tik.scope_ubuf)
        self.b2x1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2x1", scope=tik.scope_ubuf)
        self.b2x2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2x2", scope=tik.scope_ubuf)
        self.b2y1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2y1", scope=tik.scope_ubuf)
        self.b2y2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2y2", scope=tik.scope_ubuf)

        # func: create for the calculation cache of db1x/db1y/db1w/db1h in dbboxes
        self.db1x = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1x", scope=tik.scope_ubuf)
        self.db1y = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1y", scope=tik.scope_ubuf)
        self.db1w = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1w", scope=tik.scope_ubuf)
        self.db1h = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1h", scope=tik.scope_ubuf)

        # func: create and init for the calculation cache of db2x/db2y/db2w/db2h in dgtboxes
        self.db2x = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2x", scope=tik.scope_ubuf)
        self.db2y = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2y", scope=tik.scope_ubuf)
        self.db2w = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2w", scope=tik.scope_ubuf)
        self.db2h = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2h", scope=tik.scope_ubuf)

        # func: init for the calculation cache of db1x1/db1x2/db1y1/db1y2/db2x1/db2x2/db2y1/db2y2
        self.db1x1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1x1", scope=tik.scope_ubuf)
        self.db1x2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1x2", scope=tik.scope_ubuf)
        self.db1y1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1y1", scope=tik.scope_ubuf)
        self.db1y2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db1y2", scope=tik.scope_ubuf)
        self.db2x1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2x1", scope=tik.scope_ubuf)
        self.db2x2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2x2", scope=tik.scope_ubuf)
        self.db2y1 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2y1", scope=tik.scope_ubuf)
        self.db2y2 = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="db2y2", scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(Constant.BLOCK, self.db1x1, 0, Constant.REP_TIME, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, self.db1x2, 0, Constant.REP_TIME, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, self.db1y1, 0, Constant.REP_TIME, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, self.db1y2, 0, Constant.REP_TIME, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, self.db2x1, 0, Constant.REP_TIME, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, self.db2x2, 0, Constant.REP_TIME, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, self.db2y1, 0, Constant.REP_TIME, 1, 1)
        self.tik_instance.vector_dup(Constant.BLOCK, self.db2y2, 0, Constant.REP_TIME, 1, 1)

        # func: init for the calculation cache of zero
        self.tmp_zero = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="tmp_zero",
                                                 scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(Constant.BLOCK, self.tmp_zero, 0.0, Constant.REP_TIME, 1, 1)

        # func: init for the calculation cache of temp obj
        self.tmp_a = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="tmp_a", scope=tik.scope_ubuf)
        self.tmp_b = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="tmp_b", scope=tik.scope_ubuf)
        self.tmp_c = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="tmp_c", scope=tik.scope_ubuf)
        self.tmp_d = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="tmp_d", scope=tik.scope_ubuf)

        # func: init for the calculation cache of dxlen/dylen
        self.dxlen = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="dxlen", scope=tik.scope_ubuf)
        self.dylen = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="dylen", scope=tik.scope_ubuf)

        # func: init for the calculation cache of dinter/dunion/denclose
        self.dinter = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="dinter", scope=tik.scope_ubuf)
        self.dunion = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="dunion", scope=tik.scope_ubuf)
        self.denclose = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="denclose",
                                                 scope=tik.scope_ubuf)

        # func: init for the calculation cache of xlen_min/ylen_min/xlen_max/ylen_max
        self.xlen_min = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="xlen_min",
                                                 scope=tik.scope_ubuf)
        self.ylen_min = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="ylen_min",
                                                 scope=tik.scope_ubuf)
        self.xlen_max = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="xlen_max",
                                                 scope=tik.scope_ubuf)
        self.ylen_max = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="ylen_max",
                                                 scope=tik.scope_ubuf)

        # func: init for the scalar obj of 0.5
        self.half = self.tik_instance.Scalar(self.dtype, init_value=0.5)

        # func: ainitpply for the calculation cache of dy_ub/bboxes_ub/gtboxes_ub/dbboxes_ub/dgtboxes_ub
        self.dy_ub = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="targets_ub",
                                              scope=tik.scope_ubuf)
        self.bboxes_ub = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH, Constant.BOX_LOC],
                                                  name="bboxes_ub",
                                                  scope=tik.scope_ubuf)
        self.gtboxes_ub = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH, Constant.BOX_LOC],
                                                   name="gtboxes_ub",
                                                   scope=tik.scope_ubuf)
        self.dbboxes_ub = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH, Constant.BOX_LOC],
                                                   name="dbboxes_ub",
                                                   scope=tik.scope_ubuf)
        self.dgtboxes_ub = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH, Constant.BOX_LOC],
                                                    name="dgtboxes_ub",
                                                    scope=tik.scope_ubuf)

    def move_in(self, task_idx):
        """move_in"""
        # func: for dy
        self.tik_instance.data_move(self.dy_ub, self.dy[task_idx * Constant.MINI_BATCH], 0, 1, Constant.REP_TIME, 0, 0)

        # func: for bboxes
        self.tik_instance.data_move(self.b1x, self.bboxes[task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)
        self.tik_instance.data_move(self.b1y, self.bboxes[self.all_num + task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)
        self.tik_instance.data_move(self.b1w, self.bboxes[self.all_num * 2 + task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)
        self.tik_instance.data_move(self.b1h, self.bboxes[self.all_num * 3 + task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)

        # func: for gtboxes
        self.tik_instance.data_move(self.b2x, self.gtboxes[task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)
        self.tik_instance.data_move(self.b2y, self.gtboxes[self.all_num + task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)
        self.tik_instance.data_move(self.b2w, self.gtboxes[self.all_num * 2 + task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)
        self.tik_instance.data_move(self.b2h, self.gtboxes[self.all_num * 3 + task_idx * Constant.MINI_BATCH], 0, 1,
                                    Constant.REP_TIME, 0, 0)

    def data_trans(self):
        """data_trans"""
        b1w_half = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1w_half", scope=tik.scope_ubuf)
        b1h_half = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1h_half", scope=tik.scope_ubuf)

        b2w_half = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2w_half", scope=tik.scope_ubuf)
        b2h_half = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2h_half", scope=tik.scope_ubuf)

        self.tik_instance.vec_muls(Constant.BLOCK, b1w_half, self.b1w, self.half, Constant.REP_TIME, 1, 1)
        self.tik_instance.vec_muls(Constant.BLOCK, b1h_half, self.b1h, self.half, Constant.REP_TIME, 1, 1)

        # func: `b1x1 = b1x - b1w/2`
        self.tik_instance.vec_sub(Constant.BLOCK, self.b1x1, self.b1x, b1w_half, Constant.REP_TIME, 1, 1, 1)
        # func: `b1x2 = b1x + b1w/2`
        self.tik_instance.vec_add(Constant.BLOCK, self.b1x2, self.b1x, b1w_half, Constant.REP_TIME, 1, 1, 1)
        # func: `b1y1 = b1y - b1h/2`
        self.tik_instance.vec_sub(Constant.BLOCK, self.b1y1, self.b1y, b1h_half, Constant.REP_TIME, 1, 1, 1)
        # func: `b1y2 = b1y + b1h/2`
        self.tik_instance.vec_add(Constant.BLOCK, self.b1y2, self.b1y, b1h_half, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_muls(Constant.BLOCK, b2w_half, self.b2w, self.half, Constant.REP_TIME, 1, 1)
        self.tik_instance.vec_muls(Constant.BLOCK, b2h_half, self.b2h, self.half, Constant.REP_TIME, 1, 1)

        # func: `b2x1 = b2x - b2w/2`
        self.tik_instance.vec_sub(Constant.BLOCK, self.b2x1, self.b2x, b2w_half, Constant.REP_TIME, 1, 1, 1)
        # func: `b2x2 = b2x + b2w/2`
        self.tik_instance.vec_add(Constant.BLOCK, self.b2x2, self.b2x, b2w_half, Constant.REP_TIME, 1, 1, 1)
        # func: `b2y1 = b2y - b2h/2`
        self.tik_instance.vec_sub(Constant.BLOCK, self.b2y1, self.b2y, b2h_half, Constant.REP_TIME, 1, 1, 1)
        # func: `b2y2 = b2y + b2h/2`
        self.tik_instance.vec_add(Constant.BLOCK, self.b2y2, self.b2y, b2h_half, Constant.REP_TIME, 1, 1, 1)

    def update_part(self):
        """update_part"""
        b1_area = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b1_area", scope=tik.scope_ubuf)
        b2_area = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="b2_area", scope=tik.scope_ubuf)

        xmax = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="xmax", scope=tik.scope_ubuf)
        xmin = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="xmin", scope=tik.scope_ubuf)
        ymax = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="ymax", scope=tik.scope_ubuf)
        ymin = self.tik_instance.Tensor(self.dtype, [Constant.MINI_BATCH], name="ymin", scope=tik.scope_ubuf)

        # func: for inter
        self.tik_instance.vec_max(Constant.BLOCK, xmax, self.b1x1, self.b2x1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_max(Constant.BLOCK, ymax, self.b1y1, self.b2y1, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_min(Constant.BLOCK, xmin, self.b1x2, self.b2x2, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_min(Constant.BLOCK, ymin, self.b1y2, self.b2y2, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_sub(Constant.BLOCK, self.xlen_min, xmin, xmax, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.ylen_min, ymin, ymax, Constant.REP_TIME, 1, 1, 1)

        # func: choose the positive one
        with self.tik_instance.for_range(0, Constant.MINI_BATCH // Constant.MASK_BLOCK) as idx:
            self.tik_instance.vec_cmpv_gt(self.mask, self.xlen_min[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE,
                                          Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.xlen_min[Constant.MASK_BLOCK * idx], self.mask,
                                      self.xlen_min[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)

            self.tik_instance.vec_cmpv_gt(self.mask, self.ylen_min[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE,
                                          Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.ylen_min[Constant.MASK_BLOCK * idx], self.mask,
                                      self.ylen_min[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)

        # func: `inter = max(min(b1x2. b2x2) - max(b1x1, b2x1), 0) * max(min(b1y2. b2y2) - max(b1y1, b2y1), 0)`
        self.tik_instance.vec_mul(Constant.BLOCK, self.inter, self.xlen_min, self.ylen_min, Constant.REP_TIME, 1, 1, 1)

        # func: `for union, union = b1_area * b2_area - inter`
        self.tik_instance.vec_mul(Constant.BLOCK, b1_area, self.b1w, self.b1h, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_mul(Constant.BLOCK, b2_area, self.b2w, self.b2h, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_add(Constant.BLOCK, self.union, b1_area, b2_area, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.union, self.union, self.inter, Constant.REP_TIME, 1, 1, 1)

        # func: `enclose = (max(b1x2. b2x2) - min(b1x1, b2x1)) * (max(b1y2. b2y2) - min(b1y1, b2y1))`
        self.tik_instance.vec_min(Constant.BLOCK, xmin, self.b1x1, self.b2x1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_min(Constant.BLOCK, ymin, self.b1y1, self.b2y1, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_max(Constant.BLOCK, xmax, self.b1x2, self.b2x2, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_max(Constant.BLOCK, ymax, self.b1y2, self.b2y2, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_sub(Constant.BLOCK, self.xlen_max, xmax, xmin, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.ylen_max, ymax, ymin, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_mul(Constant.BLOCK, self.enclose, self.xlen_max, self.ylen_max,
                                  Constant.REP_TIME, 1, 1, 1)

    def update_dpart(self):
        """update_dpart"""
        # `for dunion, dunion = (1 / enclose - inter / (union ** 2)) * dy`
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_a, self.dy_ub, self.enclose,
                               Constant.REP_TIME, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_b, self.inter, self.union, Constant.REP_TIME,
                               1, 1, 1, 1, 1, 1)
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_c, self.tmp_b, self.union, Constant.REP_TIME,
                               1, 1, 1, 1, 1, 1)
        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_b, self.dy_ub, self.tmp_c, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.dunion, self.tmp_a, self.tmp_b, Constant.REP_TIME, 1, 1, 1)

        # `for dinter, dinter = 1 / union * dy - dunion`
        self.tik_instance.vdiv(Constant.BLOCK, self.dinter, self.dy_ub, self.union, Constant.REP_TIME, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.dinter, self.dinter, self.dunion, Constant.REP_TIME, 1, 1, 1)

        # `for denclose, denclose = -(union / (enclose ** 2)) * dy`
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_a, self.union, self.enclose,
                               Constant.REP_TIME, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vdiv(Constant.BLOCK, self.tmp_b, self.tmp_a, self.enclose,
                               Constant.REP_TIME, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_c, self.dy_ub, self.tmp_b, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.denclose, self.tmp_zero, self.tmp_c, Constant.REP_TIME, 1, 1, 1)

    def inter_part(self):
        """inter_part"""
        # for inter part
        self.tik_instance.vec_mul(Constant.BLOCK, self.dxlen, self.dinter, self.ylen_min,
                                  Constant.REP_TIME, 1, 1, 1)  # min_x
        self.tik_instance.vec_mul(Constant.BLOCK, self.dylen, self.dinter, self.xlen_min,
                                  Constant.REP_TIME, 1, 1, 1)  # min_y

        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_a, self.tmp_zero, self.dxlen,
                                  Constant.REP_TIME, 1, 1, 1)  # max_x
        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_b, self.tmp_zero, self.dylen,
                                  Constant.REP_TIME, 1, 1, 1)  # max_y
        tmp_mask = self.tik_instance.Tensor("uint16", [Constant.BLOCK], name="tmp_mask",
                                            scope=tik.scope_ubuf) # tmp_mask
        with self.tik_instance.for_range(0, Constant.MINI_BATCH // Constant.MASK_BLOCK) as idx:
            # func for max(inter1, 0) > 0
            self.tik_instance.vec_cmpv_gt(tmp_mask, self.xlen_min[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            # `func: mask_b1x2: b1x2 < b2x2, mask_b2x2 = ~mask_b1x2`
            self.tik_instance.vec_cmpv_lt(self.mask, self.b1x2[Constant.MASK_BLOCK * idx],
                                          self.b2x2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b1x2 * (inter2 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.dxlen[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b1x2 * (inter2 *dinter) * (max(inter1, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_c[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1x2[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1x2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b2x2 * (inter2 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.dxlen[Constant.MASK_BLOCK * idx],
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b2x2 * (inter2 *dinter) *(max(inter1, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_d[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2x2[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2x2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # `func: mask_b1x1: b1x1 > b2x1, mask_b2x1 = ~mask_b1x1`
            self.tik_instance.vec_cmpv_gt(self.mask, self.b1x1[Constant.MASK_BLOCK * idx],
                                          self.b2x1[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b1x1 * (-inter2 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_a[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            # func: `mask_b1x1 * (-inter2 *dinter) * (max(inter1, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_c[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1x1[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1x1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b2x1 * (-inter2 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.tmp_a[Constant.MASK_BLOCK * idx],
                                      Constant.MASK_BLOCK // Constant.MASK_BLOCK,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            # func: `mask_b2x1 * (-inter2 *dinter) * (max(inter1, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_d[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      Constant.MASK_BLOCK // Constant.MASK_BLOCK, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2x1[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2x1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func for max(inter2, 0) > 0
            self.tik_instance.vec_cmpv_gt(tmp_mask, self.ylen_min[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            # `func: mask_b1y2: b1y2 < b2y2, mask_b2y2 = ~mask_b1y2`
            self.tik_instance.vec_cmpv_lt(self.mask, self.b1y2[Constant.MASK_BLOCK * idx],
                                          self.b2y2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b1y2 * (inter1 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.dylen[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b1y2 * (inter1 *dinter) * (max(inter2, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_c[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1y2[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1y2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b2y2 * (inter1 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.dylen[Constant.MASK_BLOCK * idx],
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b2y2 * (inter1 *dinter) * (max(inter2, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_d[Constant.MASK_BLOCK * idx], self.tmp_zero,
                                      1, Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2y2[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2y2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            # `func: mask_b1y1: b1y1 > b2y1, mask_b2y1 = ~mask_b1y1`
            self.tik_instance.vec_cmpv_gt(self.mask, self.b1y1[Constant.MASK_BLOCK * idx],
                                          self.b2y1[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b1y1 * (-inter1 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_b[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            # func: `mask_b1y1 * (-inter1 *dinter) * (max(inter2, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_c[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1y1[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b2y1 * (-inter1 *dinter)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.tmp_b[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            # func: `mask_b2y1 * (-inter1 *dinter) * (max(inter2, 0) > 0)`
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], tmp_mask,
                                      self.tmp_d[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2y1[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

    def union_part(self):
        """union_part"""
        # for union part
        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_a, self.b1x2, self.b1x1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_b, self.b1y2, self.b1y1, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_a, self.tmp_a, self.dunion, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_b, self.tmp_b, self.dunion, Constant.REP_TIME, 1, 1, 1)

        # `for union part : b1x2-b1x1`
        self.tik_instance.vec_add(Constant.BLOCK, self.db1x2, self.db1x2, self.tmp_b, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.db1x1, self.db1x1, self.tmp_b, Constant.REP_TIME, 1, 1, 1)

        # `for union part : b1y2-b1y1`
        self.tik_instance.vec_add(Constant.BLOCK, self.db1y2, self.db1y2, self.tmp_a, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.db1y1, self.db1y1, self.tmp_a, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_c, self.b2x2, self.b2x1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_d, self.b2y2, self.b2y1, Constant.REP_TIME, 1, 1, 1)

        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_c, self.tmp_c, self.dunion, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_mul(Constant.BLOCK, self.tmp_d, self.tmp_d, self.dunion, Constant.REP_TIME, 1, 1, 1)

        # `for union part : b2x2-b2x1`
        self.tik_instance.vec_add(Constant.BLOCK, self.db2x2, self.db2x2, self.tmp_d, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.db2x1, self.db2x1, self.tmp_d, Constant.REP_TIME, 1, 1, 1)

        # `for union part : b2y2-b2y1`
        self.tik_instance.vec_add(Constant.BLOCK, self.db2y2, self.db2y2, self.tmp_c, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_sub(Constant.BLOCK, self.db2y1, self.db2y1, self.tmp_c, Constant.REP_TIME, 1, 1, 1)

    def enclose_part(self):
        """enclose_part"""
        # for enclose part
        self.tik_instance.vec_mul(Constant.BLOCK, self.dxlen, self.denclose, self.ylen_max,
                                  Constant.REP_TIME, 1, 1, 1)  # max_x
        self.tik_instance.vec_mul(Constant.BLOCK, self.dylen, self.denclose, self.xlen_max,
                                  Constant.REP_TIME, 1, 1, 1)  # max_y

        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_a, self.tmp_zero, self.dxlen,
                                  Constant.REP_TIME, 1, 1, 1)  # min_x
        self.tik_instance.vec_sub(Constant.BLOCK, self.tmp_b, self.tmp_zero, self.dylen,
                                  Constant.REP_TIME, 1, 1, 1)  # min_y

        with self.tik_instance.for_range(0, Constant.MINI_BATCH // Constant.MASK_BLOCK) as idx:
            # `for enclose part : max(b1_x2, b2_x2)`
            self.tik_instance.vec_cmpv_gt(self.mask, self.b1x2[Constant.MASK_BLOCK * idx],
                                          self.b2x2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)

            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.dxlen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1x2[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1x2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.dxlen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2x2[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2x2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            # `for enclose part : min(b1_x1, b2_x1)`
            self.tik_instance.vec_cmpv_lt(self.mask, self.b1x1[Constant.MASK_BLOCK * idx],
                                          self.b2x1[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_a[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1x1[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1x1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.tmp_a[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2x1[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2x1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            # `for enclose part : max(b1_y2, b2_y2)`
            self.tik_instance.vec_cmpv_gt(self.mask, self.b1y2[Constant.MASK_BLOCK * idx],
                                          self.b2y2[Constant.MASK_BLOCK * idx], 1,
                                          Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.dylen[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1y2[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1y2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.dylen[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2y2[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2y2[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

            # `for enclose part : min(b1_y1, b2_y1)`
            self.tik_instance.vec_cmpv_lt(self.mask, self.b1y1[Constant.MASK_BLOCK * idx],
                                          self.b2y1[Constant.MASK_BLOCK * idx],
                                          1, Constant.REP_STRIDE, Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_c[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_b[Constant.MASK_BLOCK * idx], self.tmp_zero, 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db1y1[Constant.MASK_BLOCK * idx],
                                      self.tmp_c[Constant.MASK_BLOCK * idx],
                                      self.db1y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_sel(Constant.MASK_BLOCK, 0, self.tmp_d[Constant.MASK_BLOCK * idx], self.mask,
                                      self.tmp_zero,
                                      self.tmp_b[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE,
                                      Constant.REP_STRIDE)
            self.tik_instance.vec_add(Constant.MASK_BLOCK, self.db2y1[Constant.MASK_BLOCK * idx],
                                      self.tmp_d[Constant.MASK_BLOCK * idx],
                                      self.db2y1[Constant.MASK_BLOCK * idx], 1, Constant.REP_STRIDE,
                                      Constant.REP_STRIDE, Constant.REP_STRIDE)

    def update_dboxes(self, task_idx):
        """update_dboxes"""
        # for b1x b1y b2x b2y
        self.tik_instance.vec_add(Constant.BLOCK, self.db1x, self.db1x1, self.db1x2, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_add(Constant.BLOCK, self.db1y, self.db1y1, self.db1y2, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_add(Constant.BLOCK, self.db2x, self.db2x1, self.db2x2, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_add(Constant.BLOCK, self.db2y, self.db2y1, self.db2y2, Constant.REP_TIME, 1, 1, 1)

        # for b1w b1h b2w b2h
        self.tik_instance.vec_sub(Constant.BLOCK, self.db1w, self.db1x2, self.db1x1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_muls(Constant.BLOCK, self.db1w, self.db1w, self.half, Constant.REP_TIME, 1, 1)

        self.tik_instance.vec_sub(Constant.BLOCK, self.db1h, self.db1y2, self.db1y1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_muls(Constant.BLOCK, self.db1h, self.db1h, self.half, Constant.REP_TIME, 1, 1)

        self.tik_instance.vec_sub(Constant.BLOCK, self.db2w, self.db2x2, self.db2x1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_muls(Constant.BLOCK, self.db2w, self.db2w, self.half, Constant.REP_TIME, 1, 1)

        self.tik_instance.vec_sub(Constant.BLOCK, self.db2h, self.db2y2, self.db2y1, Constant.REP_TIME, 1, 1, 1)
        self.tik_instance.vec_muls(Constant.BLOCK, self.db2h, self.db2h, self.half, Constant.REP_TIME, 1, 1)

        if self.move_flag:
            self.tik_instance.data_move(self.dbboxes_[task_idx * Constant.MINI_BATCH], self.db1x, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align + task_idx * Constant.MINI_BATCH],
                                        self.db1y, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align * 2 + task_idx * Constant.MINI_BATCH],
                                        self.db1w, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dbboxes_[self.all_num_align * 3 + task_idx * Constant.MINI_BATCH],
                                        self.db1h, 0, 1,
                                        Constant.REP_TIME, 0, 0)

            self.tik_instance.data_move(self.dgtboxes_[task_idx * Constant.MINI_BATCH], self.db2x, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align + task_idx * Constant.MINI_BATCH],
                                        self.db2y, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align * 2 + task_idx * Constant.MINI_BATCH],
                                        self.db2w, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dgtboxes_[self.all_num_align * 3 + task_idx * Constant.MINI_BATCH],
                                        self.db2h, 0, 1,
                                        Constant.REP_TIME, 0, 0)
        else:
            self.tik_instance.data_move(self.dbboxes[task_idx * Constant.MINI_BATCH], self.db1x, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num + task_idx * Constant.MINI_BATCH], self.db1y, 0, 1,
                                        Constant.REP_TIME,
                                        0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num * 2 + task_idx * Constant.MINI_BATCH],
                                        self.db1w, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dbboxes[self.all_num * 3 + task_idx * Constant.MINI_BATCH],
                                        self.db1h, 0, 1,
                                        Constant.REP_TIME, 0, 0)

            self.tik_instance.data_move(self.dgtboxes[task_idx * Constant.MINI_BATCH], self.db2x, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num + task_idx * Constant.MINI_BATCH],
                                        self.db2y, 0, 1, Constant.REP_TIME,
                                        0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num * 2 + task_idx * Constant.MINI_BATCH],
                                        self.db2w, 0, 1,
                                        Constant.REP_TIME, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[self.all_num * 3 + task_idx * Constant.MINI_BATCH],
                                        self.db2h, 0, 1,
                                        Constant.REP_TIME, 0, 0)

    def move_out(self):
        """move_out"""
        dbboxes_tmp = self.tik_instance.Tensor(self.dtype, [self.all_num_align], name="dbboxes_tmp",
                                               scope=tik.scope_ubuf)
        dgtboxes_tmp = self.tik_instance.Tensor(self.dtype, [self.all_num_align], name="dgtboxes_tmp",
                                                scope=tik.scope_ubuf)
        # func: Address fallback for dbboxes
        with self.tik_instance.for_range(0, Constant.BOX_LOC, thread_num=2) as idx:
            self.tik_instance.data_move(dbboxes_tmp, self.dbboxes_[idx * self.all_num_align], 0, 1,
                                        self.move_rep, 0, 0)
            self.tik_instance.data_move(self.dbboxes[idx * self.all_num], dbboxes_tmp, 0, 1, self.move_rep, 0, 0)
        # func: Address fallback for dgtboxes
        with self.tik_instance.for_range(0, Constant.BOX_LOC, thread_num=2) as idx:
            self.tik_instance.data_move(dgtboxes_tmp, self.dgtboxes_[idx * self.all_num_align], 0, 1,
                                        self.move_rep, 0, 0)
            self.tik_instance.data_move(self.dgtboxes[idx * self.all_num], dgtboxes_tmp, 0, 1, self.move_rep, 0, 0)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def giou_grad(dy, bboxes, gtboxes, dbboxes, dgtboxes, trans=False, is_cross=True, mode="iou",
              kernel_name="giou_grad"):
    """
    calculating data
    Modify : 2021-06-23

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
    op_obj = GIoUGrad(dy, bboxes, gtboxes, trans, is_cross, mode, kernel_name)

    return op_obj.compute()
