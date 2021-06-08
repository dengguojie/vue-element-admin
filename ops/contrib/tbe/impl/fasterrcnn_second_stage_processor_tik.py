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
    import decode_for_second as dc
    import softmax as sm
    import cliptowindow as clipwin
    from image_detection.image_detection import ImageDetection, Param
elif TBE_VERSION == "C7x":
    from . import decode_for_second as dc
    from . import softmax as sm
    from . import cliptowindow as clipwin
    from .image_detection import ImageDetection, Param


def _ceil_div_offline(value, factor):
    return (value + factor - 1) // factor


def _filter_greater_than(tik_instance, score_thresh, score_to_filter):
    """
    function: score_to_filter > score_thresh ? score_to_filter : 0
    @param [in] tik_instance: tik handle
    @param [in] score_thresh: thresh of score
    @param [in] score_to_filter:
    --------
    output: score_to_filter
    """

    thresh_ub = tik_instance.Tensor("float16", (score_to_filter.shape[0], 1),
                                    name="thresh_ub", scope=tik.scope_ubuf)

    left_data = score_to_filter.shape[0] % 128
    repeat_time = score_to_filter.shape[0] // 128

    with tik_instance.for_range(0, repeat_time) as iter_t:
        # use for mask less than 128 only
        tik_instance.vector_dup(128, thresh_ub[iter_t * 128, 0], score_thresh, 1, 1, 8)
        cmp_mask = tik_instance.vcmp_ge(128, score_to_filter[iter_t * 128, 0],
                                        thresh_ub[iter_t * 128, 0], 1, 1)  # cmpmask
        zeros_ub = tik_instance.Tensor("float16", (128, 1), name="zeros_ub", scope=tik.scope_ubuf)
        tik_instance.vector_dup(128, zeros_ub[0], 0, 1, 1, 8)  # use for mask less than 128 only
        tik_instance.vsel(128, 0, score_to_filter[iter_t * 128, 0], cmp_mask,
                          score_to_filter[iter_t * 128, 0], zeros_ub[0], 1, 1, 1, 1, 1, 1)

    if left_data > 0:
        tik_instance.vector_dup(left_data, thresh_ub[score_to_filter.shape[0] - left_data],
                                score_thresh, 1, 1, 8)  # use for mask less than 128 only
        cmp_mask = tik_instance.vcmp_ge(left_data, score_to_filter[score_to_filter.shape[0] - left_data],
                                        thresh_ub[score_to_filter.shape[0] - left_data], 1, 1)  # cmpmask
        zeros_ub = tik_instance.Tensor("float16", (left_data, 1), name="zeros_ub", scope=tik.scope_ubuf)
        tik_instance.vector_dup(left_data, zeros_ub[0], 0, 1, 1, 8)  # use for mask less than 128 only
        tik_instance.vsel(left_data, 0, score_to_filter[score_to_filter.shape[0] - left_data],
                          cmp_mask, score_to_filter[score_to_filter.shape[0] - left_data],
                          zeros_ub[0], 1, 1, 1, 1, 1, 1)


def _change_coodinate_frame(tik_instance, y_min_ub, x_min_ub, y_max_ub, x_max_ub, y_scale, x_scale):
    """
    function: Normalized coordinates
    @param [in] y_min_ub: y coordinate of upper left corner
    @param [in] x_min_ub: x coordinate of upper left corner
    @param [in] y_max_ub: y coordinate of lower right corner
    @param [in] x_max_ub: x coordinate of lower right corner
    @param [in] y_scale: height of image
    @param [in] x_scale: width of image
    --------
    output: y_min_ub, x_min_ub, y_max_ub, x_max_ub
    """

    y_scale = (1.0 / y_scale)
    x_scale = (1.0 / x_scale)
    shape_1 = y_min_ub.shape[1] * y_min_ub.shape[0]  # total date size
    repeat = shape_1 // 128  # repeat times
    left_data_index = repeat * 128
    left_data_mask = shape_1 - left_data_index
    tik_instance.vmuls(128, y_min_ub[0], y_min_ub[0], y_scale, repeat, 1, 1, 8, 8, 0)
    if left_data_mask:
        tik_instance.vmuls(left_data_mask, y_min_ub[left_data_index],
                           y_min_ub[left_data_index], y_scale, 1, 1, 1, 8, 8, 0)

    tik_instance.vmuls(128, x_min_ub[0], x_min_ub[0], x_scale, repeat, 1, 1, 8, 8, 0)
    if left_data_mask:
        tik_instance.vmuls(left_data_mask, x_min_ub[left_data_index],
                           x_min_ub[left_data_index], x_scale, 1, 1, 1, 8, 8, 0)

    tik_instance.vmuls(128, y_max_ub[0], y_max_ub[0], y_scale, repeat, 1, 1, 8, 8, 0)
    if left_data_mask:
        tik_instance.vmuls(left_data_mask, y_max_ub[left_data_index],
                           y_max_ub[left_data_index], y_scale, 1, 1, 1, 8, 8, 0)

    tik_instance.vmuls(128, x_max_ub[0], x_max_ub[0], x_scale, repeat, 1, 1, 8, 8, 0)
    if left_data_mask:
        tik_instance.vmuls(left_data_mask, x_max_ub[left_data_index],
                           x_max_ub[left_data_index], y_scale, 1, 1, 1, 8, 8, 0)


def _check_arg_and_get_case(shape_0, shape_1, shape_2):
    print(shape_0, shape_1, shape_2)
    if shape_0 == (1, 100, 4) and shape_1 == (100, 90 * 4) and shape_2 == (100, 91):
        return 0
    if shape_0 == (1, 300, 4) and shape_1 == (300, 8) and shape_2 == (300, 3):
        return 1
    raise RuntimeError("inputs are not valid")


class SecondStageProcessor:
    """
    Parameters
    ----------
    function_description : do Non-Maximum Suppression according to the score and prediction box,
                           and output the prediction results of each class
    scorelist : dict shape dtype format of scorelist
    boxlist : dict shape dtype format of boxlist
    proposal : dict shape dtype format of proposal
    output : dict shape and dtype of output
    kernel_name : kernel name, default value is "second_stage_processor"
    Returns
    -------
    """

    def __init__(self, box, proposal, scorelist, output, kernel_name="secondstageprocessor"):

        self.out_shape = output.get('shape')
        if self.out_shape != (1, 100, 8):
            raise RuntimeError("output shape should be (1, 100, 8)")

        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.proposal_shape = proposal.get('shape')
        self.box_shape = box.get('shape')
        self.scorelist_shape = scorelist.get('shape')
        self.case_num = _check_arg_and_get_case(self.proposal_shape, self.box_shape, self.scorelist_shape)
        if self.case_num == 0:
            self.box_shape = (128, 360)
            self.scorelist_shape = (112, 91)
        self.algn_proposal = _ceil_div_offline(self.proposal_shape[1], 16) * 16
        self.aligned_box = _ceil_div_offline(self.box_shape[0], 16) * 16
        self.aligned_out_box = _ceil_div_offline(self.out_shape[1], 16) * 16
        self.class_num = self.scorelist_shape[1] - 1

        self._basic_config()

        self.proposal_boxes_gm = self.tik_instance.Tensor("float16", self.proposal_shape,
                                                          name="proposal_boxes_gm", scope=tik.scope_gm)
        self.box_encodings_gm = self.tik_instance.Tensor("float16", self.box_shape, name="box_encodings_gm",
                                                         scope=tik.scope_gm)
        self.scorelist_gm = self.tik_instance.Tensor("float16", self.scorelist_shape,
                                                     name="scorelist_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor("float16", self.out_shape, name="score_out_gm", scope=tik.scope_gm)

        if TBE_VERSION == "C7x":
            self.output_class = self.tik_instance.Tensor("float16", (100,), name="output_class", scope=tik.scope_gm)
            self.output_boxes = self.tik_instance.Tensor("float16", (1, 100, 4), name="output_boxes",
                                                         scope=tik.scope_gm)
            self.output_num_detection = self.tik_instance.Tensor("int32", (1,), name="output_num_detection",
                                                                 scope=tik.scope_gm)

            self.output_score = self.tik_instance.Tensor("float16", (100,), name="output_score", scope=tik.scope_gm)

    def _second_stage_get_output_list(self):
        if TBE_VERSION == "C7x":
            outputs_list = [self.out_gm, self.output_class, self.output_boxes, self.output_num_detection,
                            self.output_score]
        elif TBE_VERSION == "C3x":
            outputs_list = [self.out_gm]
        return outputs_list

    def second_stage_compute(self):
        self.tik_instance = self.tik_instance
        proposal_all_classes = self.tik_instance.Tensor("float16", (1, self.class_num * self.algn_proposal, 8),
                                                        name="proposal_all_classes", scope=tik.scope_cbuf)
        proposal_all_classes_label = self.tik_instance.Tensor("float16", (1, self.class_num * self.algn_proposal, 8),
                                                              name="proposal_all_classes_label", scope=tik.scope_cbuf)
        self.proposals_sorted_cbuf = self.tik_instance.Tensor("float16", (1, 100, 8), name="proposals_sorted_cbuf",
                                                              scope=tik.scope_cbuf)  # box and score
        self.proposals_sorted_cbuf_label = self.tik_instance.Tensor(
            "float16", (1, 100, 8), name="proposals_sorted_cbuf_label", scope=tik.scope_cbuf)  # label score
        # step one , decode\softmax
        detection_handle = ImageDetection(self.tik_instance)
        with self.tik_instance.new_stmt_scope():
            self.ty_ub = self.tik_instance.Tensor("float16", (self.t_ub_size, self.aligned_box), name="ty_ub",
                                                  scope=tik.scope_ubuf)
            self.tx_ub = self.tik_instance.Tensor("float16", (self.t_ub_size, self.aligned_box), name="tx_ub",
                                                  scope=tik.scope_ubuf)
            self.th_ub = self.tik_instance.Tensor("float16", (self.t_ub_size, self.aligned_box), name="th_ub",
                                                  scope=tik.scope_ubuf)
            self.tw_ub = self.tik_instance.Tensor("float16", (self.t_ub_size, self.aligned_box), name="tw_ub",
                                                  scope=tik.scope_ubuf)
            ty_ub_slice, tx_ub_slice, th_ub_slice, tw_ub_slice, score_slice, proposal_in, proposal_out, softmax_out = \
                self._second_stage_compute_local_tensor()
            # decode
            with self.tik_instance.new_stmt_scope():
                self._scope_fasterrcnn_secondstage_postprocessor_step_one()
            # softmax
            with self.tik_instance.new_stmt_scope():
                sm.softmax(self.tik_instance, self.scorelist_gm, softmax_out)
            with self.tik_instance.new_stmt_scope():
                clipwin.clip_to_window(self.tik_instance, self.ty_ub, self.tx_ub, self.th_ub, self.tw_ub, self.list)
            # change coordinate frame
            with self.tik_instance.new_stmt_scope():
                _change_coodinate_frame(self.tik_instance, self.ty_ub, self.tx_ub, self.th_ub, self.tw_ub,
                                        self.image_y_scale, self.image_x_scale)
            with self.tik_instance.for_range(0, self.class_num) as loop:
                self._gen_each_cls_data(ty_ub_slice, tx_ub_slice, th_ub_slice, tw_ub_slice, score_slice,
                                        softmax_out, loop)
                # filtergreaterthan
                with self.tik_instance.new_stmt_scope():
                    _filter_greater_than(self.tik_instance, self.score_thresh, score_slice)
                # build proposal
                self._build_proposal(proposal_in, ty_ub_slice, tx_ub_slice, th_ub_slice, tw_ub_slice, score_slice)
                # call NMS
                self._nms(detection_handle, proposal_in, proposal_out)
                # concatenate
                self.tik_instance.data_move(proposal_all_classes[0, loop * self.algn_proposal, 0],
                                            proposal_out, 0, 1, self.algn_proposal * 8 // 16, 1, 1)
                # gen label tensor to sort
                self._process_label_to_sort(proposal_out, proposal_all_classes_label, loop)
            # all class topk after nms
        self._all_class_sort_together(detection_handle, proposal_all_classes, proposal_all_classes_label)

        # build data
        with self.tik_instance.new_stmt_scope():
            self._scope_fasterrcnn_secondstage_postprocessor_step_two()

        outputs_list = self._second_stage_get_output_list()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.box_encodings_gm, self.proposal_boxes_gm, self.scorelist_gm],
                                   outputs=outputs_list, enable_l2=False)
        return self.tik_instance

    def _scope_fasterrcnn_secondstage_postprocessor_step_one(self):
        # tiling data for decode
        proposal_boxes_ub = self.tik_instance.Tensor("float16", (1, self.algn_proposal, 4),
                                                     name="proposal_boxes_ub", scope=tik.scope_ubuf)  # map_1
        if self.case_num == 0:
            x_y_dim_zero = 96
            with self.tik_instance.for_range(0, 7) as j:  # traspose,put data into ty,tx,th,tw
                box_encodings_ub = self.tik_instance.Tensor("float16", (1, 128 // 8, 96, 4), name="box_encodings_ub",
                                                            scope=tik.scope_ubuf)  # squeeze2
                with self.tik_instance.for_range(0, 16) as i:
                    self.tik_instance.data_move(box_encodings_ub[0, i, 0, 0], self.box_encodings_gm[16 * j + i, 0],
                                                0, 1, 92 * 4 // 16, 0, 0)
                self._vnchwconv_list(self.case_num, box_encodings_ub, j)
        else:
            x_y_dim_zero = 2
            box_encodings_ub = self.tik_instance.Tensor("float16", (1, 304, 16),
                                                        name="box_encodings_ub", scope=tik.scope_ubuf)  # squeeze2
            with self.tik_instance.for_range(0, 300) as mov_t:
                self.tik_instance.data_move(box_encodings_ub[0, mov_t, 0],
                                            self.box_encodings_gm[mov_t, 0], 0, 1, 1, 0, 0)
            self._vnchwconv_list(self.case_num, box_encodings_ub)

        self.tik_instance.data_move(proposal_boxes_ub, self.proposal_boxes_gm,
                                    0, 1, self.proposal_shape[1] * 4 // 16, 0, 0)

        y_min_a = self.tik_instance.Tensor("float16", (1, self.aligned_box), name="y_min_a", scope=tik.scope_ubuf)
        x_min_a = self.tik_instance.Tensor("float16", (1, self.aligned_box), name="x_min_a", scope=tik.scope_ubuf)
        y_max_a = self.tik_instance.Tensor("float16", (1, self.aligned_box), name="y_max_a", scope=tik.scope_ubuf)
        x_max_a = self.tik_instance.Tensor("float16", (1, self.aligned_box), name="x_max_a", scope=tik.scope_ubuf)

        y_min_ub = self.tik_instance.Tensor("float16", (x_y_dim_zero, self.aligned_box),
                                            name="y_min_ub", scope=tik.scope_ubuf)
        x_min_ub = self.tik_instance.Tensor("float16", (x_y_dim_zero, self.aligned_box),
                                            name="x_min_ub", scope=tik.scope_ubuf)
        y_max_ub = self.tik_instance.Tensor("float16", (x_y_dim_zero, self.aligned_box),
                                            name="y_max_ub", scope=tik.scope_ubuf)
        x_max_ub = self.tik_instance.Tensor("float16", (x_y_dim_zero, self.aligned_box),
                                            name="x_max_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.aligned_box) as index_proposal:
            y_min_a[0, index_proposal] = proposal_boxes_ub[0, index_proposal, 0]
            x_min_a[0, index_proposal] = proposal_boxes_ub[0, index_proposal, 1]
            y_max_a[0, index_proposal] = proposal_boxes_ub[0, index_proposal, 2]
            x_max_a[0, index_proposal] = proposal_boxes_ub[0, index_proposal, 3]

        with self.tik_instance.for_range(0, x_y_dim_zero) as class_n:
            self.tik_instance.data_move(y_min_ub[class_n, 0], y_min_a[0, 0], 0, 1, self.aligned_box // 16, 0, 0)
            self.tik_instance.data_move(x_min_ub[class_n, 0], x_min_a[0, 0], 0, 1, self.aligned_box // 16, 0, 0)
            self.tik_instance.data_move(y_max_ub[class_n, 0], y_max_a[0, 0], 0, 1, self.aligned_box // 16, 0, 0)
            self.tik_instance.data_move(x_max_ub[class_n, 0], x_max_a[0, 0], 0, 1, self.aligned_box // 16, 0, 0)

        # call decode func
        dc.decode(self.tik_instance, y_min_ub, x_min_ub, y_max_ub, x_max_ub, self.ty_ub,
                  self.tx_ub, self.th_ub, self.tw_ub, self.case_num)

    def _vnchwconv_list(self, case_num, box_encodings_ub, j=0):
        if case_num == 0:
            dst_list = [self.ty_ub[0, 0 + j * 16], self.tx_ub[0, 1 + j * 16], self.th_ub[0, 2 + j * 16],
                        self.tw_ub[0, 3 + j * 16], self.ty_ub[1, 4 + j * 16], self.tx_ub[1, 5 + j * 16],
                        self.th_ub[1, 6 + j * 16], self.tw_ub[1, 7 + j * 16], self.ty_ub[2, 8 + j * 16],
                        self.tx_ub[2, 9 + j * 16], self.th_ub[2, 10 + j * 16], self.tw_ub[2, 11 + j * 16],
                        self.ty_ub[3, 12 + j * 16], self.tx_ub[3, 13 + j * 16], self.th_ub[3, 14 + j * 16],
                        self.tw_ub[3, 15 + j * 16]]

            src_list = [box_encodings_ub[0, i, 0, 0] for i in range(16)]

            self.tik_instance.vnchwconv(True, True, dst_list, src_list, 24, 32, 1)
        else:
            dst_list = [self.ty_ub[0, 0], self.tx_ub[0, 0], self.th_ub[0, 0], self.tw_ub[0, 0],
                        self.ty_ub[1, 0], self.tx_ub[1, 0], self.th_ub[1, 0], self.tw_ub[1, 0],
                        self.ty_ub[2, 0], self.tx_ub[2, 0], self.th_ub[2, 0], self.tw_ub[2, 0],
                        self.ty_ub[3, 0], self.tx_ub[3, 0], self.th_ub[3, 0], self.tw_ub[3, 0]]

            src_list = [box_encodings_ub[0, i, 0] for i in range(16)]
            self.tik_instance.vnchwconv(True, True, dst_list, src_list, 19, 1, 16)

    def _scope_fasterrcnn_secondstage_postprocessor_step_two(self):
        second_stage_out_ub = self.tik_instance.Tensor("float16", (self.aligned_out_box, self.out_shape[2]),
                                                       name="second_stage_out_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(second_stage_out_ub, self.proposals_sorted_cbuf, 0, 1,
                                    self.num_sorted_proposals * self.region_size_in_fp16 // 16, 0, 0)
        second_stage_out_ub_label = self.tik_instance.Tensor("float16", (self.aligned_out_box, self.out_shape[2]),
                                                             name="second_stage_out_ub_label", scope=tik.scope_ubuf)
        self.tik_instance.data_move(second_stage_out_ub_label, self.proposals_sorted_cbuf_label, 0, 1,
                                    self.num_sorted_proposals * self.region_size_in_fp16 // 16, 0, 0)
        second_stage_label_ub = self.tik_instance.Tensor("float16", (self.aligned_out_box,),
                                                         name="second_stage_label_ub", scope=tik.scope_ubuf)
        self.tik_instance.vextract(second_stage_label_ub, second_stage_out_ub_label, 7, 0)

        self.tik_instance.vadds(self.aligned_out_box, second_stage_label_ub, second_stage_label_ub,
                                1, 1, 1, 1, 8, 8)
        # put label into second_stage_out_ub
        with self.tik_instance.for_range(0, self.max_size_proposal) as loop:
            second_stage_out_ub[loop, 5] = second_stage_label_ub[loop]

        # move data from ub to gm
        self.tik_instance.data_move(self.out_gm, second_stage_out_ub, 0, 1,
                                    self.out_shape[1] * self.out_shape[2] // 16, 0, 0)
        if TBE_VERSION == "C7x":
            self.detection_and_parse(second_stage_out_ub)

    def detection_and_parse(self, proposals_ub):
        class_ub = self.tik_instance.Tensor("float16", (self.aligned_out_box,), name="class_ub", scope=tik.scope_ubuf)
        # extract class
        with self.tik_instance.for_range(0, self.aligned_out_box) as i:
            class_ub[i] = proposals_ub[i, 5]
        # ub -> ouputboxes
        self.tik_instance.data_move(self.output_class, class_ub, 0, 1,
                                    _ceil_div_offline(self.aligned_out_box, 16), 1, 1)
        # parse class finish here
        boxes_cols = 4
        boxes_ub = self.tik_instance.Tensor("float16", (self.aligned_out_box, boxes_cols),
                                            name="boxes_ub", scope=tik.scope_ubuf)

        # extract y1,x1,y2,x2
        with self.tik_instance.for_range(0, self.aligned_out_box) as i:
            boxes_ub[i, 0] = proposals_ub[i, 0]
            boxes_ub[i, 1] = proposals_ub[i, 1]
            boxes_ub[i, 2] = proposals_ub[i, 2]
            boxes_ub[i, 3] = proposals_ub[i, 3]
        # ub -> ouputboxes
        self.tik_instance.data_move(self.output_boxes, boxes_ub, 0, 1, _ceil_div_offline(self.out_shape[1], 4), 1, 1)
        # parse boxes finish here
        score_ub = self.tik_instance.Tensor("float16", (self.aligned_out_box,), name="score_ub", scope=tik.scope_ubuf)
        score_thresh_ub = self.tik_instance.Tensor("float16", (self.aligned_out_box,),
                                                   name="score_thresh_ub", scope=tik.scope_ubuf)
        num_ub = self.tik_instance.Tensor("float16", (16,), name="num_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.aligned_out_box, score_ub[0], 0, 1, 1, 1)
        self.tik_instance.vector_dup(self.aligned_out_box, score_thresh_ub[0], 0, 1, 1, 1)
        self.tik_instance.vector_dup(16, num_ub, 0.0, 1, 1, 1)

        # compute detection num
        for i in range(self.out_shape[1]):
            score_ub[i] = proposals_ub[i, 4]

        cmp_mask = self.tik_instance.vcmp_gt(self.out_shape[1], score_ub[0], score_thresh_ub[0], 1, 1)
        zeros_ub = self.tik_instance.Tensor("float16", (score_ub.shape[0],), name="zeros_ub", scope=tik.scope_ubuf)
        ones_ub = self.tik_instance.Tensor("float16", (score_ub.shape[0],), name="ones_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(score_ub.shape[0], zeros_ub[0], 0, 1, 1, 1)
        self.tik_instance.vector_dup(score_ub.shape[0], ones_ub[0], 1, 1, 1, 1)
        self.tik_instance.vsel(score_ub.shape[0], 0, score_ub[0], cmp_mask, ones_ub[0], zeros_ub[0], 1, 1, 1, 1, 1, 1)

        mask_scalar = self.tik_instance.Scalar("uint16", name="mask_scalar")
        ons_scalar = self.tik_instance.Scalar('float16')
        ons_scalar.set_as(1.0)
        result_ub = self.tik_instance.Tensor("float16", (16,), name="result_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(16, result_ub, 0.0, 1, 1, 1)
        with self.tik_instance.for_range(0, self.out_shape[1]) as i:
            mask_scalar.set_as(score_ub[i])
            with self.tik_instance.if_scope(mask_scalar == 15360):
                self.tik_instance.vadds(16, result_ub, result_ub, ons_scalar, 1, 1, 1, 8, 8)
        self.tik_instance.vadds(16, num_ub, result_ub, 0.0, 1, 1, 1, 8, 8)
        # ub -> ouputboxes
        self.tik_instance.data_move(self.output_num_detection, num_ub, 0, 1, 1, 0, 0)
        # parse num finish here
        # detection score
        score_ub = self.tik_instance.Tensor("float16", (self.aligned_out_box,), name="score_ub", scope=tik.scope_ubuf)
        # extract score
        with self.tik_instance.for_range(0, self.aligned_out_box) as i:
            score_ub[i] = proposals_ub[i, 4]
        # ub -> ouputboxes
        self.tik_instance.data_move(self.output_score, score_ub, 0, 1,
                                    _ceil_div_offline(self.aligned_out_box, 16), 1, 1)

    def _basic_config(self):
        self.nms_burst_num = 512
        self.down_factor = 30
        if self.case_num == 0:
            self.image_x_scale = 1024.0
            self.image_y_scale = 600.0
            self.t_ub_size = 96
            self.score_thresh = 0.3
            self.nms_thresh = 0.6
        else:
            self.image_x_scale = 1024.0
            self.image_y_scale = 320.0
            self.class_num = 2
            self.t_ub_size = 4
            self.score_thresh = 0.0
            self.nms_thresh = 0.5

        self.list = clipwin.ClipWindow()
        self.list.x = 0.
        self.list.y = 0.
        self.list.w = self.image_x_scale
        self.list.h = self.image_y_scale
        self.num_sorted_proposals = 100
        self.region_size_in_fp16 = 8
        self.max_size_proposal = 100

    def _second_stage_compute_local_tensor(self):
        # slice ub for each loop
        ty_ub_slice = self.tik_instance.Tensor("float16", (self.algn_proposal, 1), name="ty_ub_slice",
                                               scope=tik.scope_ubuf)
        tx_ub_slice = self.tik_instance.Tensor("float16", (self.algn_proposal, 1), name="tx_ub_slice",
                                               scope=tik.scope_ubuf)
        th_ub_slice = self.tik_instance.Tensor("float16", (self.algn_proposal, 1), name="th_ub_slice",
                                               scope=tik.scope_ubuf)
        tw_ub_slice = self.tik_instance.Tensor("float16", (self.algn_proposal, 1), name="tw_ub_slice",
                                               scope=tik.scope_ubuf)
        score_slice = self.tik_instance.Tensor("float16", (self.algn_proposal, 1), name="score_slice",
                                               scope=tik.scope_ubuf)

        # define proposal ub tensor for nms input\output
        proposal_in = self.tik_instance.Tensor("float16", (self.algn_proposal, 8), name="proposal_in",
                                               scope=tik.scope_ubuf)
        proposal_out = self.tik_instance.Tensor("float16", (self.algn_proposal, 8), name="proposal_out",
                                                scope=tik.scope_ubuf)
        # define input tensors and output tensors for softmax
        softmax_out = self.tik_instance.Tensor("float16",
                                               (_ceil_div_offline(self.class_num, 16) * 16, self.algn_proposal),
                                               name="softmax_out", scope=tik.scope_ubuf)
        return ty_ub_slice, tx_ub_slice, th_ub_slice, tw_ub_slice, score_slice, proposal_in, proposal_out, softmax_out

    def _gen_each_cls_data(self, ty_ub_slice, tx_ub_slice, th_ub_slice, tw_ub_slice, score_slice, softmax_out, loop):
        self.tik_instance.data_move(ty_ub_slice, self.ty_ub[loop, 0], 0, 1, self.algn_proposal // 16, 1, 1)
        self.tik_instance.data_move(tx_ub_slice, self.tx_ub[loop, 0], 0, 1, self.algn_proposal // 16, 1, 1)
        self.tik_instance.data_move(th_ub_slice, self.th_ub[loop, 0], 0, 1, self.algn_proposal // 16, 1, 1)
        self.tik_instance.data_move(tw_ub_slice, self.tw_ub[loop, 0], 0, 1, self.algn_proposal // 16, 1, 1)
        self.tik_instance.data_move(score_slice, softmax_out[loop, 0], 0, 1, self.algn_proposal // 16, 1, 1)

    def _build_proposal(self, proposal_in, ty_ub_slice, tx_ub_slice, th_ub_slice, tw_ub_slice, score_slice):
        aligned_num = self.algn_proposal - self.proposal_shape[1]
        dup_zero_ub = self.tik_instance.Tensor("float16", (aligned_num * 8,),
                                               name="dup_zero_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(aligned_num * 8, dup_zero_ub, 0, 1, 0, 0)
        self.tik_instance.vconcat(proposal_in, ty_ub_slice, self.algn_proposal // 16, 0)  # x1
        self.tik_instance.vconcat(proposal_in, tx_ub_slice, self.algn_proposal // 16, 1)  # y1
        self.tik_instance.vconcat(proposal_in, th_ub_slice, self.algn_proposal // 16, 2)  # x2
        self.tik_instance.vconcat(proposal_in, tw_ub_slice, self.algn_proposal // 16, 3)  # y2
        self.tik_instance.vconcat(proposal_in, score_slice, self.algn_proposal // 16, 4)  # score

        # clean useless data
        self.tik_instance.data_move(proposal_in[self.proposal_shape[1], 0],
                                    dup_zero_ub, 0, 1, aligned_num * 8 // 16, 0, 0)

    def _nms(self, detection_handle, proposal_in, proposal_out):
        param = Param(self.algn_proposal, self.nms_burst_num, self.num_sorted_proposals,
                      self.down_factor, self.nms_thresh, "tensorflow")
        with self.tik_instance.new_stmt_scope():
            tmp_ub_size = 32 * 1024
            mem_ub = self.tik_instance.Tensor("float16", (1, tmp_ub_size // 16, 8),
                                              name="mem_ub", scope=tik.scope_ubuf)
            proposal_in = proposal_in.reshape((1, proposal_in.shape[0], proposal_in.shape[1]))
            sorted_ub = self.tik_instance.Tensor("float16", proposal_in.shape,
                                                 name="sorted_ub", scope=tik.scope_ubuf)
            mem_intermediate = self.tik_instance.Tensor("float16", proposal_in.shape, name="mem_intermediate",
                                                        scope=tik.scope_cbuf)
            detection_handle.topk(sorted_ub, proposal_in, proposal_in.shape[1], sorted_ub.shape[1], mem_ub,
                                  mem_intermediate)
            local_tensor = detection_handle.gen_nms_local_tensor(self.algn_proposal,
                                                                 self.nms_burst_num, tik.scope_ubuf)
            proposal_out = proposal_out.reshape((1, proposal_out.shape[0], proposal_out.shape[1]))
            detection_handle.nms_after_sorted(proposal_out, sorted_ub, param, local_tensor)
            proposal_out = proposal_out.reshape((proposal_out.shape[1], proposal_out.shape[2]))

    def _process_label_to_sort(self, proposal_out, proposal_all_classes_label, loop):
        # define ub tensor for setting lable in proposal
        label_tensor_int32 = self.tik_instance.Tensor("int32", (self.algn_proposal,),
                                                      name="label_tensor_int8", scope=tik.scope_ubuf)
        label_tensor_fp16 = self.tik_instance.Tensor("float16", (self.algn_proposal,),
                                                     name="label_tensor_fp16", scope=tik.scope_ubuf)
        # store score and label
        dup_repeat = self.algn_proposal // 64
        dup_left = self.algn_proposal % 64

        if dup_repeat:
            self.tik_instance.vector_dup(64, label_tensor_int32, loop, dup_repeat, 0, 8)
            self.tik_instance.vconv(64, '', label_tensor_fp16, label_tensor_int32, dup_repeat, 0,
                                    0, 4, 8, 1.0)  # loop is int32
        if dup_left:
            self.tik_instance.vector_dup(dup_left, label_tensor_int32[dup_repeat * 64], loop, 1, 0, 0)
            self.tik_instance.vconv(dup_repeat, '', label_tensor_fp16[dup_repeat * 64],
                                    label_tensor_int32, 1, 0, 0, 4, 8, 1.0)  # int32-->fp16

        self.tik_instance.vconcat(proposal_out, label_tensor_fp16, self.algn_proposal // 16, 0)
        self.tik_instance.data_move(proposal_all_classes_label[0, loop * self.algn_proposal, 0],
                                    proposal_out, 0, 1, self.algn_proposal * 8 // 16, 1, 1)

    def _all_class_sort_together(self, detection_handle, proposal_all_classes, proposal_all_classes_label):
        # topK
        with self.tik_instance.new_stmt_scope():
            mem_intermediate = self.tik_instance.Tensor("float16", (1, self.class_num * self.algn_proposal, 8),
                                                        name="mem_intermediate", scope=tik.scope_cbuf)
            tmp_ub_size = 200 * 1024
            mem_ub = self.tik_instance.Tensor("float16", (1, tmp_ub_size // 16, 8), name="mem_ub",
                                              scope=tik.scope_ubuf)
            detection_handle.topk(self.proposals_sorted_cbuf, proposal_all_classes,
                                  self.class_num * self.algn_proposal, 100,
                                  mem_ub, mem_intermediate)
        with self.tik_instance.new_stmt_scope():
            mem_intermediate = self.tik_instance.Tensor("float16", (1, self.class_num * self.algn_proposal, 8),
                                                        name="mem_intermediate", scope=tik.scope_cbuf)
            ub_size = 200 * 1024
            mem_ub = self.tik_instance.Tensor("float16", (1, ub_size // 16, 8), name="UB", scope=tik.scope_ubuf)
            detection_handle.topk(self.proposals_sorted_cbuf_label, proposal_all_classes_label,
                                  self.class_num * self.algn_proposal, 100,
                                  mem_ub, mem_intermediate)


def fasterrcnn_second_stage_processor_tik(box, proposal, scorelist, output,
                                          out1, out2, out3, out4,
                                          kernel_name="secondstageprocessor"):
    """
    function_description : do Non-Maximum Suppression according to the score and prediction box,
                           and output the prediction results of each class
    @param [in] scorelist: Score for each box
                           dict, include shape、dtype、format
    @param [in] boxlist: refined box encodings
                         dict, include shape、dtype、format
    @param [in] proposal: proposal boxes
                          dict, include shape、dtype、format
    """

    obj = SecondStageProcessor(box, proposal, scorelist, output, kernel_name)
    obj.second_stage_compute()


def fasterrcnn_second_stage_processor_tik_c3x(box, proposal, scorelist, output, kernel_name="secondstageprocessor"):
    """
    function_description : do Non-Maximum Suppression according to the score and prediction box,
                           and output the prediction results of each class
    @param [in] scorelist: Score for each box
                           dict, include shape、dtype、format
    @param [in] boxlist: refined box encodings
                         dict, include shape、dtype、format
    @param [in] proposal: proposal boxes
                          dict, include shape、dtype、format
    """

    obj = SecondStageProcessor(box, proposal, scorelist, output, kernel_name)
    obj.second_stage_compute()
