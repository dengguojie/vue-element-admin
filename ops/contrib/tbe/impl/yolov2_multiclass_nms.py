# coding=utf-8
from te import tik
from .image_detection import ImageDetection

ONE_POINT_ZERO = 15360
PROPOSALS = 1024
N_PROPOSALS = 112
MAX_OUT_BOX_NUM = 112
NMS_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
DOWN_FACTOR = 1
PROPOSALS_SZ_FP16 = 8
PROPOSALS_SZ_BYTE = 16


def ceil_div_offline(value, factor):
    if value % factor == 0:
        return value // factor
    else:
        return value // factor + 1


def ceil_div_online(ib, value, factor):
    result = ib.Scalar(value.dtype, name="result")
    with ib.if_scope(value % factor == 0):
        result.set_as(value // factor)
    with ib.else_scope():
        result.set_as(value // factor + 1)
    return result


def get_reduced_proposal(tik_instance, out_proposal, in_proposal, _down_factor, burst_input_proposal_num):
    num_coord_addr = ceil_div_offline(burst_input_proposal_num, 128) * 128
    coord_addr = tik_instance.Tensor("float16", [4, num_coord_addr], name="coord_addr", scope=tik.scope_ubuf)

    tik_instance.vextract(coord_addr[0], in_proposal[0], coord_addr.shape[1] // 16, 0)  # x1
    tik_instance.vextract(coord_addr[coord_addr.shape[1] * 1], in_proposal[0], coord_addr.shape[1] // 16, 1)  # y1
    tik_instance.vextract(coord_addr[coord_addr.shape[1] * 2], in_proposal[0], coord_addr.shape[1] // 16, 2)  # x2
    tik_instance.vextract(coord_addr[coord_addr.shape[1] * 3], in_proposal[0], coord_addr.shape[1] // 16, 3)  # y2

    tik_instance.vmuls(128, coord_addr[0], coord_addr[0], _down_factor, coord_addr.shape[1] // num_coord_addr, 1, 1, 8,
                       8)  # x1*_down_factor
    tik_instance.vmuls(128, coord_addr[coord_addr.shape[1] * 1], coord_addr[coord_addr.shape[1] * 1], _down_factor,
                       coord_addr.shape[1] // num_coord_addr, 1, 1, 8, 8)  # y1*_down_factor
    tik_instance.vmuls(128, coord_addr[coord_addr.shape[1] * 2], coord_addr[coord_addr.shape[1] * 2], _down_factor,
                       coord_addr.shape[1] // num_coord_addr, 1, 1, 8, 8)  # x2*_down_factor
    tik_instance.vmuls(128, coord_addr[coord_addr.shape[1] * 3], coord_addr[coord_addr.shape[1] * 3], _down_factor,
                       coord_addr.shape[1] // num_coord_addr, 1, 1, 8, 8)  # y2*_down_factor

    tik_instance.vadds(128, coord_addr[0], coord_addr[0], 1, ceil_div_offline(coord_addr.shape[1], 128), 1, 1, 8,
                       8)  # x1*_down_factor+1
    tik_instance.vadds(128, coord_addr[coord_addr.shape[1] * 1], coord_addr[coord_addr.shape[1] * 1], 1,
                       ceil_div_offline(coord_addr.shape[1], 128), 1, 1, 8, 8)  # y1*_down_factor+1

    tik_instance.vconcat(out_proposal[0], coord_addr[0], coord_addr.shape[1] // 16, 0)  # x1
    tik_instance.vconcat(out_proposal[0], coord_addr[coord_addr.shape[1] * 1], coord_addr.shape[1] // 16, 1)  # y1
    tik_instance.vconcat(out_proposal[0], coord_addr[coord_addr.shape[1] * 2], coord_addr.shape[1] // 16, 2)  # x2
    tik_instance.vconcat(out_proposal[0], coord_addr[coord_addr.shape[1] * 3], coord_addr.shape[1] // 16, 3)  # y2
    return out_proposal


def sort_proposals(tik_instance, proposals_sorted_ub, in_proposals):
    detection_handle = ImageDetection(tik_instance)
    with tik_instance.new_stmt_scope():
        mem_intermediate_gm = tik_instance.Tensor("float16", (1, in_proposals.shape[1], 8), name="mem_intermediate_gm",
                                                  scope=tik.scope_cbuf)
        tmp_ub_size = 100 * 1024
        mem_ub = tik_instance.Tensor("float16", (1, tmp_ub_size // 16, 8), name="mem_ub", scope=tik.scope_ubuf)
        detection_handle.topk(proposals_sorted_ub, in_proposals, 1808, 1024, mem_ub, mem_intermediate_gm)


def non_max_suppression(tik_instance, in_proposals, out_proposals, valid_cnt, selected_num):
    """
    non_max_suppression
    :param valid_cnt:
    :param selected_num:
    :param tik_instance:
    :param in_proposals: [1, 1808, 8]
    :param out_proposals: [112, 8]
    :return:
    """
    # UB 194k  # only use formal 112 anchors
    proposals_sorted_ub = tik_instance.Tensor("float16", (1, PROPOSALS, PROPOSALS_SZ_FP16),
                                              name="proposals_sorted_ub", scope=tik.scope_ubuf)
    proposals_reduced_sorted_ub = tik_instance.Tensor("float16", (PROPOSALS, PROPOSALS_SZ_FP16),
                                                      name="proposals_reduced_sorted_ub", scope=tik.scope_ubuf)
    tmp_nms_thresh = NMS_THRESHOLD / (1 + NMS_THRESHOLD)
    tik_instance.vector_dup(128, out_proposals[0], 0, 1, 1, 8)
    tik_instance.vector_dup(128, proposals_reduced_sorted_ub[0], 0, PROPOSALS * 8 // 128, 1, 8)

    sort_proposals(tik_instance, proposals_sorted_ub, in_proposals)

    # only static shape support
    get_reduced_proposal(tik_instance, proposals_reduced_sorted_ub, proposals_sorted_ub, DOWN_FACTOR, PROPOSALS)

    proposals_iou_ub = tik_instance.Tensor("float16", (PROPOSALS, 16), name="proposals_iou_ub", scope=tik.scope_ubuf)
    proposals_add_area_and_thresh = tik_instance.Tensor("float16", (PROPOSALS, 16),
                                                        name="proposals_add_area_and_thresh",
                                                        scope=tik.scope_ubuf)  # 56.5k
    proposals_area_ub = tik_instance.Tensor("float16", (PROPOSALS, 1), name="proposals_area_ub", scope=tik.scope_ubuf)
    tik_instance.vrpac(proposals_area_ub[0], proposals_reduced_sorted_ub[0],
                       ceil_div_online(tik_instance, valid_cnt, 16))

    # suppress vector
    sup_matrix_ub = tik_instance.Tensor("uint16", (PROPOSALS,), name="sup_matrix_ub", scope=tik.scope_ubuf)  # 3.53k
    sup_vector_ub = tik_instance.Tensor("uint16", [PROPOSALS], name="sup_vector_ub", scope=tik.scope_ubuf)  # 3.75k
    tmp_sup_vector_ub = tik_instance.Tensor("uint16", [PROPOSALS], name="tmp_sup_vector_ub", scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, sup_matrix_ub[0], 1, ceil_div_offline(PROPOSALS, 128), 1, 8)
    tik_instance.vector_dup(128, sup_vector_ub[0], 1, ceil_div_offline(PROPOSALS, 128), 1, 8)
    tik_instance.vector_dup(128, tmp_sup_vector_ub[0], 1, ceil_div_offline(PROPOSALS, 128), 1, 8)

    length = tik_instance.Scalar(dtype="uint16", init_value=0)
    scalar_uint16 = tik_instance.Scalar(dtype="uint16", init_value=0)
    sup_vector_ub[0].set_as(scalar_uint16)
    with tik_instance.for_range(0, ceil_div_online(tik_instance, valid_cnt, 16)) as i:
        length.set_as(length + 16)
        tik_instance.viou(proposals_iou_ub[0, 0], proposals_reduced_sorted_ub[0, 0],
                          proposals_reduced_sorted_ub[i * 16, 0], i + 1)
        tik_instance.vaadd(proposals_add_area_and_thresh[0, 0], proposals_area_ub[0, 0], proposals_area_ub[i * 16, 0],
                           i + 1)
        tik_instance.vmuls(128, proposals_add_area_and_thresh[0, 0], proposals_add_area_and_thresh[0, 0],
                           tmp_nms_thresh, ceil_div_online(tik_instance, length, 8), 1, 1, 8, 8)
        tik_instance.vcmpv_gt(sup_matrix_ub[0], proposals_iou_ub[0, 0], proposals_add_area_and_thresh[0, 0],
                              ceil_div_online(tik_instance, length, 8), 1, 1, 8, 8)
        # update suppress vector
        rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
        with tik_instance.if_scope(i > 0):
            rpn_cor_ir = tik_instance.rpn_cor(sup_matrix_ub[0], tmp_sup_vector_ub[0], 1, 1, repeat_times=i)
        with tik_instance.else_scope():
            rpn_cor_ir = tik_instance.rpn_cor(sup_matrix_ub[0], sup_vector_ub[0], 1, 1, repeat_times=1)
        tik_instance.rpn_cor_diag(tmp_sup_vector_ub[i * 16], sup_matrix_ub[length - 16], rpn_cor_ir)

    # suppress vector to get proposal data
    tmp = tik_instance.Scalar('float16')
    selected_num.set_as(0)
    with tik_instance.for_range(0, valid_cnt) as i:
        with tik_instance.if_scope(tmp_sup_vector_ub[i] == 0):
            with tik_instance.for_range(0, 8) as j:
                tmp.set_as(proposals_sorted_ub[0, i, j])
                out_proposals[selected_num, j].set_as(tmp)
            selected_num.set_as(selected_num + 1)
