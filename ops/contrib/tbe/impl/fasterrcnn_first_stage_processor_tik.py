# -*- coding:utf-8 -*-
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

if TBE_VERSION == "C3x":
    from cliptowindow_for_firststage import ClipWindow, clip_to_window
    from decode_for_firststage import decode
    from padorclipbox import pad_or_clip_box
    from softmax_tik_first import softmax
    from image_detection.image_detection import ImageDetection, Param
elif TBE_VERSION == "C7x":
    from .cliptowindow_for_firststage import ClipWindow, clip_to_window
    from .decode_for_firststage import decode
    from .padorclipbox import pad_or_clip_box
    from .softmax_tik_first import softmax
    from .image_detection import ImageDetection, Param

BLOCKLENGTH = 1920


def ceil_div_offline(value, factor):
    return (value + factor - 1) // factor


def upper_clip(value, clipstep):
    temp = clipstep * 2
    return ((value + clipstep) // temp) * temp


class Nmspara:
    n_proposals = 29184
    max_output_box_num = 100
    n_sorted_proposals = 10000
    nms_thresh = 0.7
    down_factor = 0.3
    burst_input_proposal_num = 512


class FirstStageProcessor:
    """
    Parameters
    ----------
        function_description : withdraw foregrounds of feature mapx.
        scorelist : dict shape dtype format of scorelist
        boxlist : dict shape dtype format of boxlist
        anchorlist : dict shape dtype format of anchorlist
        output : dict shape and dtype of output
        kernel_name : kernel name, default value is "first_stage_processor"
    Returns
    -------
        None
    """

    def __init__(self, scorelist, boxlist, anchorlist, output, kernel_name='first_stage_processor'):
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        scorelist_shape = scorelist.get('shape')
        boxlist_shape = boxlist.get('shape')
        anchorlist_shape = anchorlist.get('shape')

        output_shape = output.get('shape')

        if output.get('dtype') != 'float16':
            raise RuntimeError("output data type has to be float16")
        if output_shape != (1, 300, 4) and output_shape != (1, 100, 4):
            raise RuntimeError("output data shape not match")
        if scorelist_shape == (1, 61440, 2) or scorelist_shape == (1, 1, 61440, 2):
            scorelist_shape = (1, 1, 61440, 2)
        elif scorelist_shape == (1, 29184, 2) or scorelist_shape == (1, 1, 29184, 2):
            scorelist_shape = (1, 1, 29184, 2)
        else:
            raise RuntimeError("input scorelist not valid")

        if scorelist.get('dtype') != 'float16' or boxlist.get('dtype') != 'float16' \
                or anchorlist.get('dtype') != 'float16':
            raise RuntimeError("input data type has to be float16")
        proposal_shape = (1, scorelist_shape[2], 8)
        mem_intermediate_shape = (1, scorelist_shape[2] + 3, 8)
        self.max_size_per_class = output_shape[1]
        self.max_total_size = output_shape[1]
        self.length = scorelist_shape[2]
        self.kernel_name = kernel_name
        self._basic_config()
        # input
        self.scorelist_gm = self.tik_instance.Tensor("float16", scorelist_shape, name="scorelist_gm",
                                                     scope=tik.scope_gm)
        self.boxlist_gm = self.tik_instance.Tensor("float16", boxlist_shape, name="boxlist_gm",
                                                   scope=tik.scope_gm)
        self.anchorlist_gm = self.tik_instance.Tensor("float16", anchorlist_shape, name="anchorlist_gm",
                                                      scope=tik.scope_gm)
        if TBE_VERSION is "C7x":
            self.proposal_tmp_gm = self.tik_instance.Tensor("float16", proposal_shape, name="proposal_tmp_gm",
                                                            scope=tik.scope_gm, is_workspace=True)
            self.mem_intermediate_gm = self.tik_instance.Tensor("float16", mem_intermediate_shape,
                                                                name="mem_intermediate_gm", scope=tik.scope_gm,
                                                                is_workspace=True)
        elif TBE_VERSION is "C3x":
            self.proposal_tmp_gm = self.tik_instance.Tensor("float16", proposal_shape, name="proposal_tmp_gm",
                                                            scope=tik.scope_gm)
            self.mem_intermediate_gm = self.tik_instance.Tensor("float16", mem_intermediate_shape,
                                                                name="mem_intermediate_gm", scope=tik.scope_gm)
        # output
        self.output_gm = self.tik_instance.Tensor("float16", output_shape, name="output_gm", scope=tik.scope_gm)

    def _basic_config(self):
        self.score_thresh = 0.
        self.iou_thresh = 0.7
        self.scalefactor_0 = 10.0
        self.scalefactor_1 = 10.0
        self.scalefactor_2 = 5.0
        self.scalefactor_3 = 5.0
        self.scalefactor_num = 4
        self.width = 1024.0
        self.nmspara = Nmspara()
        self.nmspara.burst_input_proposal_num = 512
        if self.length == 61440:
            self.height = 320
            self.nmspara.n_proposals = 61440
            self.nmspara.max_output_box_num = 300
            self.nmspara.n_sorted_proposals = 24576
            self.nmspara.nms_thresh = 0.7
            self.nmspara.down_factor = 0.1
        else:
            self.height = 600
            self.nmspara.n_proposals = 29184
            self.nmspara.max_output_box_num = 100
            self.nmspara.n_sorted_proposals = 10000
            self.nmspara.nms_thresh = 0.7
            self.nmspara.down_factor = 0.3
        self.list = ClipWindow()
        self.list.x = 0.
        self.list.y = 0.
        self.list.w = float(self.height)
        self.list.h = 1024.

    def tilling_mode_select(self):
        self.mode = 1

    def global_init(self):
        pass

    def first_stage_compute(self):
        # softmax->decode->cliptowindow
        with self.tik_instance.new_stmt_scope():
            # outputproposal_gm, outputbox_gm, outputanchor_gm,
            self._scope_fasterrcnn_first_stage_processor_step_1()
            # to nms
        detection_hanle = ImageDetection(self.tik_instance)
        proposals_sorted_l1 = self.tik_instance.Tensor("float16", (1, self.nmspara.n_sorted_proposals, 8),
                                                       name="proposals_sorted_l1", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            tmp_ub_size = 200 * 1024
            mem_ub = self.tik_instance.Tensor("float16", (1, tmp_ub_size // 16, 8), name="mem_ub", scope=tik.scope_ubuf)
            detection_hanle.topk(proposals_sorted_l1, self.proposal_tmp_gm, self.nmspara.n_proposals,
                                 self.nmspara.n_sorted_proposals, mem_ub, self.mem_intermediate_gm)
        output_num = ceil_div_offline(self.max_total_size, 16) * 16
        param = Param(self.nmspara.n_sorted_proposals, self.nmspara.burst_input_proposal_num,
                      output_num, self.nmspara.down_factor, self.nmspara.nms_thresh, "tensorflow")
        nmsout_ub = self.tik_instance.Tensor("float16", (1, output_num, 8), name="nmsout_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            local_tensor = detection_hanle.gen_nms_local_tensor(output_num, self.nmspara.burst_input_proposal_num,
                                                                tik.scope_ubuf)
            detection_hanle.nms_after_sorted(nmsout_ub, proposals_sorted_l1, param, local_tensor)
            nmsout_ub = nmsout_ub.reshape((output_num, 8))
        # boxlist to map1
        boxout_ub = self.tik_instance.Tensor("float16", (1, self.max_total_size, 4),
                                             name="boxout_ub", scope=tik.scope_ubuf)

        # test cliptowindows
        pad_or_clip_box(self.tik_instance, nmsout_ub, boxout_ub, self.max_total_size, 1)

        self.tik_instance.data_move(self.output_gm[0, 0, 0], boxout_ub[0, 0, 0],
                                    0, 1, self.max_total_size * 4 // 16, 1, 1)
        if TBE_VERSION is "C7x":
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.scorelist_gm, self.boxlist_gm, self.anchorlist_gm],
                                       outputs=[self.output_gm])
        elif TBE_VERSION is "C3x":
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.scorelist_gm, self.boxlist_gm, self.anchorlist_gm,
                                               self.proposal_tmp_gm, self.mem_intermediate_gm],
                                       outputs=[self.output_gm])
        return self.tik_instance

    def _scope_fasterrcnn_first_stage_processor_step_1(self):
        alloclength = (self.length // BLOCKLENGTH) * BLOCKLENGTH
        resnum = self.length - alloclength
        reslength = (resnum + 63) // 64 * 64
        iter_num = self.length // BLOCKLENGTH
        iter_num1 = BLOCKLENGTH // 64
        iter_num2 = reslength // 64

        # ub allocbuf size equals to 103KB
        boxlist_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH // 4, 16), name="boxlist_ub",
                                              scope=tik.scope_ubuf)
        anchorlist_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH // 4, 16), name="anchorlist_ub",
                                                 scope=tik.scope_ubuf)
        scorelist_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH // 4, 8), name="scorelist_ub",
                                                scope=tik.scope_ubuf)
        proposal_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH, 8), name="proposal_ub", scope=tik.scope_ubuf)
        scoreleft_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="scoreleft_ub", scope=tik.scope_ubuf)
        scoreright_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="scoreright_ub", scope=tik.scope_ubuf)
        temp_ub = self.tik_instance.Tensor("float16", (16, 16), name="boxlist_ub", scope=tik.scope_ubuf)

        # box equals to [ymin, xmin, ymax, xmax]
        ymin_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="ymin_ub", scope=tik.scope_ubuf)
        xmin_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="xmin_ub", scope=tik.scope_ubuf)
        ymax_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="ymax_ub", scope=tik.scope_ubuf)
        xmax_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="xmax_ub", scope=tik.scope_ubuf)

        # anchor equals to [ymin, xmin, ymax, xmax]
        ymin_anchor_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="ymin_anchor_ub",
                                                  scope=tik.scope_ubuf)
        xmin_anchor_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="xmin_anchor_ub",
                                                  scope=tik.scope_ubuf)
        ymax_anchor_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="ymax_anchor_ub",
                                                  scope=tik.scope_ubuf)
        xmax_anchor_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="xmax_anchor_ub",
                                                  scope=tik.scope_ubuf)
        # score list
        score_ub = self.tik_instance.Tensor("float16", (BLOCKLENGTH,), name="score_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, iter_num) as i:
            self._decode_and_score_compute(boxlist_ub, anchorlist_ub, scorelist_ub, proposal_ub, temp_ub, scoreleft_ub,
                                           scoreright_ub, ymin_ub, xmin_ub, ymax_ub, xmax_ub, ymin_anchor_ub,
                                           xmin_anchor_ub, ymax_anchor_ub, xmax_anchor_ub, score_ub, i,
                                           iter_num1, BLOCKLENGTH)

        if resnum > 0:
            self._decode_and_score_compute(boxlist_ub, anchorlist_ub, scorelist_ub, proposal_ub, temp_ub, scoreleft_ub,
                                           scoreright_ub, ymin_ub, xmin_ub, ymax_ub, xmax_ub, ymin_anchor_ub,
                                           xmin_anchor_ub, ymax_anchor_ub, xmax_anchor_ub, score_ub, iter_num,
                                           iter_num2, reslength)

    def _decode_and_score_compute(self, boxlist_ub, anchorlist_ub, scorelist_ub, proposal_ub, temp_ub, scoreleft_ub,
                                  scoreright_ub, ymin_ub, xmin_ub, ymax_ub, xmax_ub, ymin_anchor_ub, xmin_anchor_ub,
                                  ymax_anchor_ub, xmax_anchor_ub, score_ub, iter_num_i, item_num_j, process_data_len):

        self.tik_instance.data_move(boxlist_ub[0, 0], self.boxlist_gm[0, BLOCKLENGTH * iter_num_i, 0, 0],
                                    0, 1, process_data_len * 4 // 16, 1, 1)
        self.tik_instance.data_move(anchorlist_ub, self.anchorlist_gm[0, BLOCKLENGTH * iter_num_i, 0, 0],
                                    0, 1, process_data_len * 4 // 16, 1, 1)
        self.tik_instance.data_move(scorelist_ub[0, 0], self.scorelist_gm[0, 0, BLOCKLENGTH * iter_num_i, 0],
                                    0, 1, process_data_len * 2 // 16, 1, 1)

        # merger channels, 16x16, [1,4,7], iter_num_i equals to 60
        with self.tik_instance.for_range(0, item_num_j) as item_loop:
            # boxlist
            self.tik_instance.vtranspose(temp_ub[0, 0], boxlist_ub[16 * item_loop, 0])
            self.tik_instance.data_move(ymin_ub[64 * item_loop], temp_ub[0, 0], 0, 4, 1, 3, 0)
            self.tik_instance.data_move(xmin_ub[64 * item_loop], temp_ub[1, 0], 0, 4, 1, 3, 0)
            self.tik_instance.data_move(ymax_ub[64 * item_loop], temp_ub[2, 0], 0, 4, 1, 3, 0)
            self.tik_instance.data_move(xmax_ub[64 * item_loop], temp_ub[3, 0], 0, 4, 1, 3, 0)

            # anchorlist
            self.tik_instance.vtranspose(temp_ub[0, 0], anchorlist_ub[16 * item_loop, 0])
            self.tik_instance.data_move(ymin_anchor_ub[64 * item_loop], temp_ub[0, 0], 0, 4, 1, 3, 0)
            self.tik_instance.data_move(xmin_anchor_ub[64 * item_loop], temp_ub[1, 0], 0, 4, 1, 3, 0)
            self.tik_instance.data_move(ymax_anchor_ub[64 * item_loop], temp_ub[2, 0], 0, 4, 1, 3, 0)
            self.tik_instance.data_move(xmax_anchor_ub[64 * item_loop], temp_ub[3, 0], 0, 4, 1, 3, 0)

            # scorelist
            self.tik_instance.vextract(scoreleft_ub[64 * item_loop + 0], scorelist_ub[16 * item_loop, 0], 1, 0)
            self.tik_instance.vextract(scoreleft_ub[64 * item_loop + 16], scorelist_ub[16 * item_loop, 0], 1, 2)

            self.tik_instance.vextract(scoreright_ub[64 * item_loop + 0], scorelist_ub[16 * item_loop, 0], 1, 1)
            self.tik_instance.vextract(scoreright_ub[64 * item_loop + 16], scorelist_ub[16 * item_loop, 0], 1, 3)

            with self.tik_instance.for_range(0, 16) as k:
                scoreleft_ub[64 * item_loop + 32 + k] = scorelist_ub[16 * item_loop + k, 4]
                scoreleft_ub[64 * item_loop + 48 + k] = scorelist_ub[16 * item_loop + k, 6]
                scoreright_ub[64 * item_loop + 32 + k] = scorelist_ub[16 * item_loop + k, 5]
                scoreright_ub[64 * item_loop + 48 + k] = scorelist_ub[16 * item_loop + k, 7]

        # step 1: SoftMax  input1:[process_data_len]  input2:[process_data_len], output[process_data_len]
        softmax(self.tik_instance, scoreleft_ub, scoreright_ub, score_ub)

        # step 2: Decode Windows
        with self.tik_instance.new_stmt_scope():
            decode(self.tik_instance, ymin_anchor_ub, xmin_anchor_ub, ymax_anchor_ub,
                   xmax_anchor_ub, ymin_ub, xmin_ub, ymax_ub, xmax_ub)

        # step 3: ClipToWindow
        with self.tik_instance.new_stmt_scope():
            clip_to_window(self.tik_instance, ymin_anchor_ub, xmin_anchor_ub,
                           ymax_anchor_ub, xmax_anchor_ub, process_data_len, self.list)

        # step 4: proposal concat
        self.tik_instance.vconcat(proposal_ub, ymin_anchor_ub, process_data_len // 16, 0)
        self.tik_instance.vconcat(proposal_ub, xmin_anchor_ub, process_data_len // 16, 1)
        self.tik_instance.vconcat(proposal_ub, ymax_anchor_ub, process_data_len // 16, 2)
        self.tik_instance.vconcat(proposal_ub, xmax_anchor_ub, process_data_len // 16, 3)
        self.tik_instance.vconcat(proposal_ub, score_ub, process_data_len // 16, 4)

        # step 5: output l1buf
        self.tik_instance.data_move(self.proposal_tmp_gm[0, BLOCKLENGTH * iter_num_i, 0],
                                    proposal_ub[0, 0], 0, 1, process_data_len * 8 // 16, 1, 1)


def fasterrcnn_first_stage_processor_tik(scorelist, boxlist, anchorlist, output,
                                         kernel_name='first_stage_processor'):
    """
    function_description : withdraw foregrounds of feature mapx.
    @param scorelist : dict shape dtype format of scorelist
    @param boxlist : dict shape dtype format of boxlist
    @param anchorlist : dict shape dtype format of anchorlist
    @param output : dict shape and dtype of output
    @param kernel_name : kernel name, default value is "first_stage_processor"
    """
    obj = FirstStageProcessor(scorelist, boxlist, anchorlist, output, kernel_name)
    obj.first_stage_compute()
