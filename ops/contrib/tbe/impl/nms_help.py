class NMSHelp(object):
    """
    NMSHelp
    """
    def __init__(self, tik_inst):
        self.tik_inst = tik_inst
        self.boxes_size = 8

    def _ceil_div_offline(self, value, factor):
        return (value + factor - 1) // factor

    def _ceil_div_online(self, value, factor):
        result = self.tik_inst.Scalar(value.dtype, name="result")
        result.set_as((value + factor - 1) // factor)
        return result

    def _get_handing_num(self, handing_num, left_boxs_num, burst_input_num):
        with self.tik_inst.if_scope(left_boxs_num < burst_input_num):
            handing_num.set_as(left_boxs_num)
        with self.tik_inst.else_scope():
            handing_num.set_as(burst_input_num)

    def _get_reduced_boxs(self, reduced_boxes, ori_boxes, reduce_mem_ub, down_factor, input_num, model_type="caffe"):
        if input_num % 16 != 0:
            raise RuntimeError("reducing boxes num should align to 16")
        mem_ub_shape = reduce_mem_ub.shape
        if mem_ub_shape[0] != 4 or mem_ub_shape[1] < self._ceil_div_offline(input_num, 128) * 128:
            raise RuntimeError("reduce mem ub shape be [4, align_to_128(input_num)]!")

        ext_repeat_time = input_num // 16
        if ext_repeat_time > 255:
            raise RuntimeError("reducing boxes num should less than 255 * 16")

        self.tik_inst.vextract(reduce_mem_ub[0, 0], ori_boxes, ext_repeat_time, 0)  # x1
        self.tik_inst.vextract(reduce_mem_ub[1, 0], ori_boxes, ext_repeat_time, 1)  # y1
        self.tik_inst.vextract(reduce_mem_ub[2, 0], ori_boxes, ext_repeat_time, 2)  # x2
        self.tik_inst.vextract(reduce_mem_ub[3, 0], ori_boxes, ext_repeat_time, 3)  # y2

        mul_rep = input_num // 128

        self.tik_inst.vmuls(128, reduce_mem_ub[0, 0], reduce_mem_ub[0, 0], down_factor, mul_rep, 1, 1, 8,
                            8)  # x1*downFactor
        self.tik_inst.vmuls(128, reduce_mem_ub[1, 0], reduce_mem_ub[1, 0], down_factor, mul_rep, 1, 1, 8,
                            8)  # y1*downFactor
        self.tik_inst.vmuls(128, reduce_mem_ub[2, 0], reduce_mem_ub[2, 0], down_factor, mul_rep, 1, 1, 8,
                            8)  # x2*downFactor
        self.tik_inst.vmuls(128, reduce_mem_ub[3, 0], reduce_mem_ub[3, 0], down_factor, mul_rep, 1, 1, 8,
                            8)  # y2*downFactor

        if model_type == "tensorflow":
            self.tik_inst.vadds(128, reduce_mem_ub[0, 0], reduce_mem_ub[0, 0], 1, mul_rep, 1, 1, 8,
                                8)  # x1*downFactor + 1
            self.tik_inst.vadds(128, reduce_mem_ub[1, 0], reduce_mem_ub[1, 0], 1, mul_rep, 1, 1, 8,
                                8)  # y1*downFactor + 1

        self.tik_inst.vconcat(reduced_boxes, reduce_mem_ub[0, 0], ext_repeat_time, 0)  # x1
        self.tik_inst.vconcat(reduced_boxes, reduce_mem_ub[1, 0], ext_repeat_time, 1)  # y1
        self.tik_inst.vconcat(reduced_boxes, reduce_mem_ub[2, 0], ext_repeat_time, 2)  # x2
        self.tik_inst.vconcat(reduced_boxes, reduce_mem_ub[3, 0], ext_repeat_time, 3)  # y2

    def _reset_temp_ub(self, local_tensor):
        self.tik_inst.vector_dup(128, local_tensor["burst_sup_vec_ub"], 1,
                                 local_tensor["burst_sup_vec_ub"].shape[0] // 128, 1, 8)
        self.tik_inst.vector_dup(128, local_tensor["burst_reduced_boxes_ub"], 0,
                                 local_tensor["burst_reduced_boxes_ub"].shape[0] // 16, 1, 8)
        self.tik_inst.vector_dup(128, local_tensor["burst_ori_boxes_ub"], 0,
                                 local_tensor["burst_ori_boxes_ub"].shape[0] // 16, 1, 8)
        self.tik_inst.vector_dup(128, local_tensor["burst_area_ub"], 0, local_tensor["burst_area_ub"].shape[0] // 128,
                                 1, 8)
        repeat_time = local_tensor["burst_iou_ub"].shape[0] // 8
        left_num = local_tensor["burst_iou_ub"].shape[0] * local_tensor["burst_iou_ub"].shape[1] % 128
        if repeat_time > 0:
            self.tik_inst.vector_dup(128, local_tensor["burst_iou_ub"], 0, repeat_time, 1, 8)
            self.tik_inst.vector_dup(128, local_tensor["burst_added_ub"], 0, repeat_time, 1, 8)

        if left_num != 0:
            self.tik_inst.vector_dup(left_num, local_tensor["burst_iou_ub"][repeat_time * 8, 0], 0, 1, 1, 0)
            self.tik_inst.vector_dup(left_num, local_tensor["burst_added_ub"][repeat_time * 8, 0], 0, 1, 1, 0)

    def _reset_selected_tensor(self, selected_boxes_ub, selected_reduced_boxes_ub, selected_area, if_init):
        num = min(selected_reduced_boxes_ub.shape[0], selected_boxes_ub.shape[1])
        repeat_time = num // 16
        left_num = num * 8 % 128
        if repeat_time > 0:
            if if_init:
                self.tik_inst.vector_dup(128, selected_boxes_ub, 0, repeat_time, 1, 8)
            self.tik_inst.vector_dup(128, selected_reduced_boxes_ub, 0, repeat_time, 1, 8)

        if left_num != 0:
            if if_init:
                self.tik_inst.vector_dup(left_num, selected_boxes_ub[0, repeat_time * 16, 0], 0, 1, 1, 0)
            self.tik_inst.vector_dup(left_num, selected_reduced_boxes_ub[repeat_time * 16, 0], 0, 1, 1, 0)

        repeat_time_area = selected_area.shape[0] // 128
        left_num_area = selected_area.shape[0] % 128

        if repeat_time_area > 0:
            self.tik_inst.vector_dup(128, selected_area, 0, repeat_time_area, 1, 8)
        if left_num_area != 0:
            self.tik_inst.vector_dup(left_num_area, selected_area[repeat_time_area * 128], 0, 1, 1, 0)

    def _nms_for_each_burst(self, handing_num, tmp_selected_num, output_num, length, burst_iou_ub,
                            selected_reduced_boxes_ub, burst_reduced_boxes_ub, aligned_selected_num, burst_added_ub,
                            selected_area, burst_area_ub, thresh, burst_sup_matrix_ub, sup_vec_ub, burst_sup_vec_ub):
        with self.tik_inst.for_range(0, self._ceil_div_online(handing_num, 16)) as i:
            with self.tik_inst.if_scope(tmp_selected_num < output_num):
                length.set_as(length + 16)
                # calculate intersection of selected_reduced_boxs_ub and burst_reduced_boxes_ub
                self.tik_inst.viou(burst_iou_ub[0, 0],
                                   selected_reduced_boxes_ub,
                                   burst_reduced_boxes_ub[i * 16, 0],
                                   aligned_selected_num // 16)
                # calculate intersection of burst_reduced_boxes_ub and burst_reduced_boxes_ub(include itself)
                self.tik_inst.viou(burst_iou_ub[aligned_selected_num, 0],
                                   burst_reduced_boxes_ub,
                                   burst_reduced_boxes_ub[i * 16, 0], i + 1)
                # calculate join of burst_area_ub and selected_area
                self.tik_inst.vaadd(burst_added_ub, selected_area,
                                    burst_area_ub[i * 16],
                                    aligned_selected_num // 16)
                # calculate intersection of burst_area_ub and burst_area_ub(include itself)
                self.tik_inst.vaadd(burst_added_ub[aligned_selected_num, 0],
                                    burst_area_ub, burst_area_ub[i * 16], i + 1)
                # calculate join * (thresh//(1+thresh)
                self.tik_inst.vmuls(128, burst_added_ub[0, 0],
                                    burst_added_ub[0, 0], thresh,
                                    self._ceil_div_online(length, 8), 1, 1, 8, 8)
                # compare and generate suppression matrix
                self.tik_inst.vcmpv_gt(burst_sup_matrix_ub[0],
                                       burst_iou_ub[0, 0],
                                       burst_added_ub[0, 0],
                                       self._ceil_div_online(length, 8), 1, 1, 8, 8)
                # generate suppression vector
                # clear rpn_cor_ir
                rpn_cor_ir = self.tik_inst.set_rpn_cor_ir(0)
                # non-diagonal
                rpn_cor_ir = self.tik_inst.rpn_cor(burst_sup_matrix_ub[0],
                                                   sup_vec_ub[0], 1, 1,
                                                   aligned_selected_num // 16)
                with self.tik_inst.if_scope(i > 0):
                    rpn_cor_ir = self.tik_inst.rpn_cor(
                        burst_sup_matrix_ub[aligned_selected_num],
                        burst_sup_vec_ub[0], 1, 1, i)
                # diagonal
                self.tik_inst.rpn_cor_diag(burst_sup_vec_ub[i * 16],
                                           burst_sup_matrix_ub[length - 16],
                                           rpn_cor_ir)
                with self.tik_inst.for_range(0, 16) as j:
                    with self.tik_inst.if_scope(burst_sup_vec_ub[i * 16 + j] == 0):
                        tmp_selected_num.set_as(tmp_selected_num + 1)

    def _update_selected_boxes(self, selected_boxes_ub, burst_ori_boxes_ub, box_value_scalar, selected_num, index):
        with self.tik_inst.for_range(0, 5) as j:
            box_value_scalar.set_as(burst_ori_boxes_ub[index, j])
            selected_boxes_ub[0, selected_num, j].set_as(box_value_scalar)

    def _update_reduced_selected_boxes(self, tmp_selected_num, output_num, box_value_scalar, burst_reduced_boxes_ub,
                                       selected_reduced_boxes_ub, selected_num, index):
        # update selReducedProposals_ub
        with self.tik_inst.for_range(0, 5) as j:
            with self.tik_inst.if_scope(tmp_selected_num < output_num):
                box_value_scalar.set_as(burst_reduced_boxes_ub[index, j])
                selected_reduced_boxes_ub[selected_num, j].set_as(box_value_scalar)

    def _update_area_supvec(self, tmp_selected_num, output_num, box_value_scalar, burst_area_ub, selected_num,
                            selected_area, sup_vec_ub, zero_scalar, index):
        with self.tik_inst.if_scope(tmp_selected_num < output_num):
            # update selArea_ub
            box_value_scalar.set_as(burst_area_ub[index])
            selected_area[selected_num].set_as(box_value_scalar)
            # update supVec_ub
            sup_vec_ub[selected_num].set_as(zero_scalar)

    def _update_selected_info(self, handing_num, selected_num, output_num, burst_sup_vec_ub, box_value_scalar,
                              burst_ori_boxes_ub, selected_boxes_ub, tmp_selected_num, burst_reduced_boxes_ub,
                              selected_reduced_boxes_ub, burst_area_ub, selected_area, sup_vec_ub, zero_scalar,
                              index_output, index_start):
        # find & move unsuppressed proposals
        with self.tik_inst.for_range(0, handing_num) as i:
            with self.tik_inst.if_scope(selected_num < output_num):
                with self.tik_inst.if_scope(burst_sup_vec_ub[i] == 0):
                    self._update_selected_boxes(selected_boxes_ub, burst_ori_boxes_ub, box_value_scalar,
                                                selected_num, i)
                    self._index_add(index_output, selected_num, index_start, i)
                    # update selReducedProposals_ub
                    self._update_reduced_selected_boxes(tmp_selected_num, output_num, box_value_scalar,
                                                        burst_reduced_boxes_ub, selected_reduced_boxes_ub,
                                                        selected_num,
                                                        i)
                    self._update_area_supvec(tmp_selected_num, output_num, box_value_scalar, burst_area_ub,
                                             selected_num, selected_area, sup_vec_ub, zero_scalar, i)
                    # update counter
                    selected_num.set_as(selected_num + 1)

    def _index_add(self, index_output, selected_num, index_start, i):
        if index_output is not None:
            index_output[selected_num].set_as(index_start + i)

    def gen_local_tensor(self, output_num, burst_num, scope):
        """
        :param output_num:
        :param burst_num:
        :param scope:
        :return:
        """
        local_tensor = {}
        output_num_align_to_128 = self._ceil_div_offline(output_num, 128) * 128
        output_num_align_to_16 = self._ceil_div_offline(output_num, 16) * 16
        local_tensor["selected_reduced_boxes_ub"] = \
            self.tik_inst.Tensor("float16", (output_num_align_to_16, self.boxes_size),
                                 name="selected_reduced_boxes_ub",
                                 scope=scope)
        local_tensor["selected_area"] = \
            self.tik_inst.Tensor("float16", (output_num_align_to_16,), name="selected_area", scope=scope)
        local_tensor["sup_vec_ub"] = \
            self.tik_inst.Tensor("uint16", (output_num_align_to_128,), name="sup_vec_ub", scope=scope)

        local_tensor["burst_ori_boxes_ub"] = \
            self.tik_inst.Tensor("float16", (burst_num, self.boxes_size), name="burst_ori_boxes_ub",
                                 scope=scope)
        local_tensor["burst_reduced_boxes_ub"] = \
            self.tik_inst.Tensor("float16", (burst_num, self.boxes_size), name="burst_reduced_boxes_ub",
                                 scope=scope)
        local_tensor["burst_area_ub"] = \
            self.tik_inst.Tensor("float16", (burst_num,), name="burst_area_ub", scope=scope)
        local_tensor["burst_iou_ub"] = \
            self.tik_inst.Tensor("float16", (output_num_align_to_16 + burst_num, 16), name="burst_iou_ub", scope=scope)
        local_tensor["burst_added_ub"] = \
            self.tik_inst.Tensor("float16", (output_num_align_to_16 + burst_num, 16), name="burst_added_ub",
                                 scope=scope)
        local_tensor["burst_reduced_mem_ub"] = \
            self.tik_inst.Tensor("float16", (4, burst_num), name="burst_reduced_mem_ub", scope=scope)
        local_tensor["burst_sup_matrix_ub"] = \
            self.tik_inst.Tensor("uint16", (output_num_align_to_16 + burst_num,), name="burst_sup_matrix_ub",
                                 scope=scope)
        local_tensor["burst_sup_vec_ub"] = \
            self.tik_inst.Tensor("uint16", (burst_num,), name="burst_sup_vec_ub", scope=scope)
        return local_tensor

    def _update_len_and_select_num(self, length, selected_num, tmp_selected_num):
        length.set_as(self._ceil_div_online(selected_num, 16) * 16)
        tmp_selected_num.set_as(selected_num)
        aligned_selected_num = self._ceil_div_online(selected_num, 16) * 16
        return aligned_selected_num

    def _init_left_boxes_num(self, left_boxes_num, param, valid_box_num):
        with self.tik_inst.if_scope(valid_box_num >= 0):
            left_boxes_num.set_as(valid_box_num)
        with self.tik_inst.else_scope():
            left_boxes_num.set_as(param.input_num)

    def _check_param(self, param):
        if param.input_num % 16 != 0:
            raise RuntimeError("input num {} must align to 16"
                               .format(param.input_num))
        if param.burst_num % 128 != 0:
            raise RuntimeError("burst num {} must align to 128"
                               .format(param.burst_num))
        if param.max_output_num % 2 != 0:
            raise RuntimeError("max output num {} must align to 2"
                               .format(param.max_output_num))

    def _deal_odd(self, burst_ori_boxes_ub, handing_num, if_odd):
        if if_odd:
            with self.tik_inst.if_scope(handing_num % 2 == 1):
                self.tik_inst.vector_dup([0, 3840], burst_ori_boxes_ub[handing_num // 2 * 2, 0],
                                         0.0, 1, 1, 8)
                self.tik_inst.vector_dup([0, 4096], burst_ori_boxes_ub[handing_num // 2 * 2, 0],
                                         -65504.0, 1, 1, 8)

    def nms_after_sorted(self, selected_boxes_ub, sorted_boxes, param, local_tensor, valid_box_num,
                         index_nms, if_init, if_odd):
        """
        non maximum suppression for sorted boxes
        parameters
        :param selected_boxes_ub: output tensor, shape(1, max_output_num, 8), float16, scope_ub
        :param sorted_boxes:input tensor, shape(1, N, 8), float16, scope ub l1 or gm
        :param param: Param class
        :param local_tensor: local tensor
        :param valid_box_num: the num of valid box
        :param index_nms: the index of boxes
        :param if_init: init selected_boxes_ub 0
        :param if_odd: if deal odd num
        :return:
        """
        self._check_param(param)
        input_burst = param.burst_num  # size align to 128
        output_num = param.max_output_num  # align to 16
        down_factor, model_type = param.down_factor, param.model_type
        thresh = param.nms_thresh / (1 + param.nms_thresh)
        # local tensor
        selected_reduced_boxes_ub = local_tensor["selected_reduced_boxes_ub"]  # shape equal to selected boxes ub
        selected_area = local_tensor["selected_area"]  # shape output num
        sup_vec_ub = local_tensor["sup_vec_ub"]  # output align to 128

        self.tik_inst.vector_dup(128, sup_vec_ub, 1, sup_vec_ub.shape[0] // 128, 1, 8)

        burst_ori_boxes_ub = local_tensor["burst_ori_boxes_ub"]
        burst_sup_vec_ub = local_tensor["burst_sup_vec_ub"]  # size align to 128
        burst_reduced_boxes_ub = local_tensor["burst_reduced_boxes_ub"]  # shape (burst_num, 8)
        burst_area_ub = local_tensor["burst_area_ub"]  # shape (burst_num, 8)
        burst_iou_ub = local_tensor["burst_iou_ub"]  # shape (output_num, 8)
        burst_added_ub = local_tensor["burst_added_ub"]  # shape (output_num + burst_num)
        burst_sup_matrix_ub = local_tensor["burst_sup_matrix_ub"]  # shape (output_num + burst_num), type: uint16
        burst_reduced_mem_ub = local_tensor["burst_reduced_mem_ub"]  # shape (burst_num align to 128), type: uint16

        # local scalar
        selected_num = self.tik_inst.Scalar("uint16", init_value=0)
        left_boxes_num = self.tik_inst.Scalar("int32")
        self._init_left_boxes_num(left_boxes_num, param, valid_box_num)
        handing_num = self.tik_inst.Scalar("int32", init_value=0)
        zero_scalar = self.tik_inst.Scalar("uint16", init_value=0)
        tmp_selected_num = self.tik_inst.Scalar("uint16", init_value=0)

        length = self.tik_inst.Scalar("uint32")
        box_value_scalar = self.tik_inst.Scalar("float16")
        # check
        sup_vec_ub[0].set_as(zero_scalar)
        self._reset_selected_tensor(selected_boxes_ub, selected_reduced_boxes_ub, selected_area, if_init)
        num_burst = self._ceil_div_online(left_boxes_num, input_burst)
        with self.tik_inst.for_range(0, num_burst) as burst_index:
            with self.tik_inst.if_scope(selected_num < output_num):
                self._get_handing_num(handing_num, left_boxes_num, input_burst)
                # clear burst sup vec
                self._reset_temp_ub(local_tensor)
                # get burst boxes from topK result
                self.tik_inst.data_move(burst_ori_boxes_ub, sorted_boxes[burst_index * input_burst * 8], 0, 1,
                                        self._ceil_div_online(handing_num, 2), 0, 0)
                self._deal_odd(burst_ori_boxes_ub, handing_num, if_odd)
                # reduce fresh proposal
                self._get_reduced_boxs(burst_reduced_boxes_ub, burst_ori_boxes_ub, burst_reduced_mem_ub,
                                       down_factor, input_burst, model_type)
                # calculate the area of reduced-proposals
                self.tik_inst.vrpac(burst_area_ub, burst_reduced_boxes_ub, self._ceil_div_online(handing_num, 16))
                # start to update iou and or area from the first 16 proposal and get suppression
                # vector 16 by 16 proposal
                aligned_selected_num = self._update_len_and_select_num(length, selected_num, tmp_selected_num)
                self._nms_for_each_burst(handing_num, tmp_selected_num, output_num, length, burst_iou_ub,
                                         selected_reduced_boxes_ub, burst_reduced_boxes_ub, aligned_selected_num,
                                         burst_added_ub, selected_area, burst_area_ub, thresh, burst_sup_matrix_ub,
                                         sup_vec_ub, burst_sup_vec_ub)
                # find & mov unsuppressed proposals
                self._update_selected_info(handing_num, selected_num, output_num, burst_sup_vec_ub, box_value_scalar,
                                           burst_ori_boxes_ub, selected_boxes_ub, tmp_selected_num,
                                           burst_reduced_boxes_ub, selected_reduced_boxes_ub, burst_area_ub,
                                           selected_area, sup_vec_ub, zero_scalar, index_nms,
                                           input_burst * burst_index)
            left_boxes_num.set_as(left_boxes_num - handing_num)
        return selected_num
