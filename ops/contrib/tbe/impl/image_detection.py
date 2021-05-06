from .topk_help import TopkHelp
from .nms_help import NMSHelp


class Param(object):
    def __init__(self, input_num, burst_num, max_output_num, down_factor, nms_thresh, model_type):
        self.input_num = input_num
        self.burst_num = burst_num
        self.max_output_num = max_output_num
        self.down_factor = down_factor
        self.nms_thresh = nms_thresh
        self.model_type = model_type


class ImageDetection(object):
    def __init__(self, tik_instance):
        self.tik_inst = tik_instance
        self.topk_help_handle = TopkHelp(self.tik_inst)
        self.nms_help_handle = NMSHelp(self.tik_inst)
        self.nms_valid_box_num_scalar = self.tik_inst.Scalar("int32", init_value=-1)

    def topk(self, boxes_sorted, boxes_original, boxes_num, n_required, ub_buffer, mem_intermediate):
        """
        Select top K elements from last dimension
        :param boxes_sorted: output tensor, (1, n_proposals, 8) , float16, scope ub l1 or gm
        :param boxes_original: (1, n_required, 8), float16, scope ub l1 or gm
        :param boxes_num: ori proposals num.
        :param n_required: Number of largest elements to be select
        :param ub_buffer: mem tensor. (1, ub_size // 16, 8), float16, scope ub tensor
        :param mem_intermediate: Shape(1, >=n_proposals, 8), float16, scope ub l1 or gm
        :return: local tensor
        """
        self.topk_help_handle.tik_topk(boxes_sorted, boxes_original, boxes_num, n_required, ub_buffer, mem_intermediate)

    def gen_nms_local_tensor(self, max_output_num, burst_num, scope):
        """
        help malloc nms local tensor
        :param max_output_num: max output boxes
        :param burst_num: burst processor boxes number
        :param scope: tik.scope_ubuf
        :return: local tensor
        """
        return self.nms_help_handle.gen_local_tensor(max_output_num, burst_num, scope)

    def nms_after_sorted(self, selected_boxes_ub, sorted_boxes, param, local_tensor,
                         index_nms=None, if_init=True, if_odd=False):
        """
        non maximum suppression for sorted boxes
        parameters
        :param selected_boxes_ub: output tensor, (1, max_output_num, 8), float16
        :param sorted_boxes: (1, N, 8) float16, scope up l1 or gm
        :param param:
        :param local_tensor: local tensor
        :param index_nms: the index of boxes
        :param if_init: init selected boxes_ub 0
        :param if_odd: deal odd num
        :return: output valid num
        """
        return self.nms_help_handle.nms_after_sorted(selected_boxes_ub, sorted_boxes, param, local_tensor,
                                                     self.nms_valid_box_num_scalar, index_nms, if_init, if_odd)

    def set_nms_valid_box_num(self, scalar_val_int32):
        """
        set nms valid box num
        :param
        scalar_val_int32: Scalar, type int32
        :return:
        """
        if scalar_val_int32.dtype != self.nms_valid_box_num_scalar.dtype:
            raise RuntimeError("scalar_val_int32 type {} must be int32"
                               .format(scalar_val_int32.dtype))
        self.nms_valid_box_num_scalar.set_as(scalar_val_int32)
