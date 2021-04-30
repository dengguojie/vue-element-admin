# coding=utf-8
from te import tik
from .yolov2_decode import decode
from .yolov2_score import class_score
from .yolov2_multiclass_nms import non_max_suppression

MASK_128 = 128
NMS_CLASSES = 90
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MAX_BOX_PER_CLASS = 100
MAX_BOX_PER_CLASS_ALGIN2BLOCK = 112
MAX_TOTAL_DETECTIONS = 100
MAX_TOTAL_BOX_ALIGN2BLOCK = 112
PROPOSAL_NUM = 1805
PROPOSAL_NUM_ALGIN2BLOCK = 1808
MAX_VALID_PROPOSALS = 1024


# Window
class Window:
    def __init__(self):
        self.y_min = 0.0
        self.x_min = 0.0
        self.y_max = 1.0
        self.x_max = 1.0


class GBuf:
    def __init__(self, tik_instance):
        self.tik_instance = tik_instance

        # gm variable
        self.box_encoding_gm = tik_instance.Tensor("float16", (1, 1808, 16), name="box_encoding_gm", scope=tik.scope_gm)
        self.box_shadow_gm = self.box_encoding_gm[0, :, :]
        self.anchor_data = tik_instance.Tensor("float16", (1808, 4), name="anchor_data", scope=tik.scope_gm)
        self.scores_gm = tik_instance.Tensor("float16", (1, 1808, ceil_div_offline(NMS_CLASSES, 16) * 16),
                                             name="scores_gm", scope=tik.scope_gm)
        self.boxout_gm = tik_instance.Tensor("float16", (MAX_TOTAL_DETECTIONS, 8), name="out_gm", scope=tik.scope_gm)

        # proposal gather variable 112 -> ceil_div_offline(MAX_TOTAL_DETECTIONS, 16)*16     ceil_div_offline(
        # max_detections_per_class, 16)*16
        self.scores_l1 = tik_instance.Tensor("float16", (1, ceil_div_offline(NMS_CLASSES, 16) * 16, 1808),
                                             name="scores_l1", scope=tik.scope_cbuf)
        self.decode_l1 = tik_instance.Tensor("float16", (1, 4, 1808), name="decode_l1", scope=tik.scope_cbuf)

        # define ub tensor for setting lable in proposal
        self.label_tensor_fp16 = tik_instance.Tensor("float16", (128, 1), name="label_tensor_fp16",
                                                     scope=tik.scope_ubuf)

        # define proposal ub tensor for nms input\output
        self.proposal_ub = tik_instance.Tensor("float16", (1, 1808, 8), name="proposal_ub",
                                               scope=tik.scope_ubuf)  # (1,1808,8)
        self.proposals_out = tik_instance.Tensor("float16", (128, 8), name="proposals_out",
                                                 scope=tik.scope_ubuf)  # (1,1808,8)
        self.gather_out_ub = tik_instance.Tensor("float16", (128, 8), name="gather_out_ub",
                                                 scope=tik.scope_ubuf)  # (1,1808,8)

        self.tik_instance.vector_dup(MASK_128, self.label_tensor_fp16[0, 0], 0, 128 // 128, dst_blk_stride=1,
                                     dst_rep_stride=8)
        self.tik_instance.vector_dup(MASK_128, self.proposal_ub, 0, (1808 * 8) // 128, dst_blk_stride=1,
                                     dst_rep_stride=8)
        self.tik_instance.vector_dup(MASK_128, self.proposals_out, 0, (128 * 8) // 128, dst_blk_stride=1,
                                     dst_rep_stride=8)
        self.tik_instance.vector_dup(MASK_128, self.gather_out_ub, 0, (128 * 8) // 128, dst_blk_stride=1,
                                     dst_rep_stride=8)


def ceil_div_offline(value, factor):
    if value % factor == 0:
        return value // factor
    else:
        return value // factor + 1


def clip_to_window(tik_instance, y_min, x_min, y_max, x_max, length, window):
    """
    clip coordinate to window
    :param tik_instance:
    :param y_min:
    :param x_min:
    :param y_max:
    :param x_max:
    :param length:
    :param window:
    :return:
    """
    win_y_min = tik_instance.Tensor("float16", (length,), name="win_y_min", scope=tik.scope_ubuf)
    win_x_min = tik_instance.Tensor("float16", (length,), name="win_x_min", scope=tik.scope_ubuf)
    win_y_max = tik_instance.Tensor("float16", (length,), name="win_y_max", scope=tik.scope_ubuf)
    win_x_max = tik_instance.Tensor("float16", (length,), name="win_x_max", scope=tik.scope_ubuf)

    clip_y_min = tik_instance.Scalar("float16")
    clip_x_min = tik_instance.Scalar("float16")
    clip_y_max = tik_instance.Scalar("float16")
    clip_x_max = tik_instance.Scalar("float16")

    clip_y_min.set_as(window.y_min)
    clip_x_min.set_as(window.x_min)
    clip_y_max.set_as(window.y_max)
    clip_x_max.set_as(window.x_max)

    # duplicate clipwindow shape to vector
    tik_instance.vector_dup(MASK_128, win_y_min, clip_y_min, length // 128, 1, 8, 0)
    tik_instance.vector_dup(MASK_128, win_x_min, clip_x_min, length // 128, 1, 8, 0)
    tik_instance.vector_dup(MASK_128, win_y_max, clip_y_max, length // 128, 1, 8, 0)
    tik_instance.vector_dup(MASK_128, win_x_max, clip_x_max, length // 128, 1, 8, 0)

    # y_min_clipped eq tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
    tik_instance.vmin(MASK_128, y_min, y_min, win_y_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(MASK_128, y_min, y_min, win_y_min, length // 128, 1, 1, 1, 8, 8, 8, 0)

    # y_max_clipped eq tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
    tik_instance.vmin(MASK_128, y_max, y_max, win_y_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(MASK_128, y_max, y_max, win_y_min, length // 128, 1, 1, 1, 8, 8, 8, 0)

    # x_min_clipped eq tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
    tik_instance.vmin(MASK_128, x_min, x_min, win_x_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(MASK_128, x_min, x_min, win_x_min, length // 128, 1, 1, 1, 8, 8, 8, 0)

    # x_max_clipped eq tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
    tik_instance.vmin(MASK_128, x_max, x_max, win_x_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(MASK_128, x_max, x_max, win_x_min, length // 128, 1, 1, 1, 8, 8, 8, 0)


def filter_greater_than(tik_instance, score_thresh, scores_filter, n_proposals, valid_cnt):
    """
    filter out scores that are lower than threshold
    two cases
    1.len > 128, for and tail
    2.len <= 128, once operator
    :param tik_instance:
    :param score_thresh:
    :param scores_filter:
    :param n_proposals:
    :param valid_cnt:
    :return:valid_cnt
    optimization：by filtering score,valid anchor ->1024
    """
    score_vector_ub = tik_instance.Tensor("float16", (128,), name="score_vector_ub", scope=tik.scope_ubuf)
    thresh_ub = tik_instance.Tensor("float16", (128,), name="thresh_ub", scope=tik.scope_ubuf)  # set thresh vector
    ones_ub = tik_instance.Tensor("float16", (128,), name="zeros_ub", scope=tik.scope_ubuf)
    zeros_ub = tik_instance.Tensor("float16", (128,), name="zeros_ub", scope=tik.scope_ubuf)
    tik_instance.vector_dup(MASK_128, thresh_ub[0], score_thresh, repeat_times=1, dst_blk_stride=1,
                            dst_rep_stride=8)  # use for mask less than 1808 only
    tik_instance.vector_dup(MASK_128, ones_ub[0], 1., 1, 1, 1)
    tik_instance.vector_dup(MASK_128, zeros_ub[0], 0., repeat_times=1, dst_blk_stride=1,
                            dst_rep_stride=8)  # use for mask less than 1808 only

    valid_cnt.set_as(0)
    mask_scalar = tik_instance.Scalar("uint16", name="mask_scalar")
    repeat_times = n_proposals // 128
    with tik_instance.for_range(0, repeat_times) as i:
        cmp_mask = tik_instance.vcmp_ge(MASK_128, scores_filter[0 + 128 * i], thresh_ub[0], 1, 1)  # cmpmask
        tik_instance.vsel(MASK_128, 0, score_vector_ub[0], cmp_mask, ones_ub[0], zeros_ub[0], 1, 1, 1, 1, 1, 1)
        with tik_instance.for_range(0, MASK_128) as bit_i:
            mask_scalar.set_as(score_vector_ub[bit_i])
            with tik_instance.if_scope(mask_scalar == 15360):  # 1.0
                valid_cnt.set_as(valid_cnt + 1)
        tik_instance.vsel(MASK_128, 0, scores_filter[0 + 128 * i], cmp_mask, scores_filter[0 + 128 * i], zeros_ub[0], 1,
                          1, 1, 1, 1, 1, 1)
    # tail data
    with tik_instance.if_scope(n_proposals % 128 != 0):
        tail_length = n_proposals % 128
        cmp_mask = tik_instance.vcmp_ge(tail_length, scores_filter[0 + 128 * repeat_times], thresh_ub[0], 1,
                                        1)  # cmpmask
        tik_instance.vsel(tail_length, 0, score_vector_ub[0], cmp_mask, ones_ub[0], zeros_ub[0], 1, 1, 1, 1, 1, 1)
        with tik_instance.for_range(0, tail_length) as bit_i:
            mask_scalar.set_as(score_vector_ub[bit_i])
            with tik_instance.if_scope(mask_scalar == 15360):  # 1.0
                valid_cnt.set_as(valid_cnt + 1)
        tik_instance.vsel(tail_length, 0, scores_filter[0 + 128 * repeat_times], cmp_mask,
                          scores_filter[0 + 128 * repeat_times], zeros_ub[0], 1, 1, 1, 1, 1, 1, 1)


def transpose_tensor(tik_instance, data_gm, data_l1):
    """
    transpose tensor and fits by data_l1 shape
    :param tik_instance:
    :param data_gm:[1, 1808, 96], [1, 1808, 16]
    :param data_l1: [1, 90, 1808], [1, 4, 1808]
    :return:
    """
    dst_ub = tik_instance.Tensor("float16", (16, 1808), name="dst_ub", scope=tik.scope_ubuf)
    src_ub = tik_instance.Tensor("float16", (1808, 16), name="src_ub", scope=tik.scope_ubuf)

    with tik_instance.for_range(0, ceil_div_offline(data_gm.shape[2], 16)) as loopi:
        tik_instance.data_move(src_ub, data_gm[0, 0, loopi * 16], 0, nburst=data_gm.shape[1], burst=1,
                               src_stride=ceil_div_offline(data_gm.shape[2], 16) - 1, dst_stride=0)
        dst_list = [dst_ub[0, 0], dst_ub[1, 0], dst_ub[2, 0], dst_ub[3, 0], dst_ub[4, 0], dst_ub[5, 0], dst_ub[6, 0],
                    dst_ub[7, 0], dst_ub[8, 0], dst_ub[9, 0], dst_ub[10, 0], dst_ub[11, 0], dst_ub[12, 0],
                    dst_ub[13, 0], dst_ub[14, 0], dst_ub[15, 0]]
        src_list = [src_ub[0, 0], src_ub[1, 0], src_ub[2, 0], src_ub[3, 0], src_ub[4, 0], src_ub[5, 0], src_ub[6, 0],
                    src_ub[7, 0], src_ub[8, 0], src_ub[9, 0], src_ub[10, 0], src_ub[11, 0], src_ub[12, 0],
                    src_ub[13, 0], src_ub[14, 0], src_ub[15, 0]]
        tik_instance.vnchwconv(True, True, dst_list, src_list, data_gm.shape[1] // 16, 1, 16)
        with tik_instance.if_scope(loopi == ceil_div_offline(data_gm.shape[2], 16) - 1):
            tik_instance.data_move(data_l1[0, loopi * 16, 0], dst_ub[0], 0, nburst=1,
                                   burst=(data_l1.shape[1] - loopi * 16) * data_l1.shape[2] // 16, src_stride=0,
                                   dst_stride=0)
        with tik_instance.else_scope():
            tik_instance.data_move(data_l1[0, loopi * 16, 0], dst_ub[0], 0, nburst=1, burst=16 * data_l1.shape[2] // 16,
                                   src_stride=0, dst_stride=0)


def pad_last_dim(tik_instance, box_endcoded_gm, decode_l1):
    """
    pad last dimension to 16
    :param tik_instance:
    :param box_endcoded_gm: [1, 1808, 4]
    :param decode_l1: [1, 1808, 16]
    :return:
    """
    box_temp = tik_instance.Tensor("float16", (1808, 4), name="box_temp", scope=tik.scope_ubuf)
    box_dst = tik_instance.Tensor("float16", (1808, 16), name="box_temp", scope=tik.scope_ubuf)
    tik_instance.data_move(box_temp[0, 0], box_endcoded_gm[0, 0], 0, 1, 4 * 1808 // 16, 0, 0)
    tik_instance.vector_dup(MASK_128, box_dst, 0, (1808 * 16) // 128, dst_blk_stride=1, dst_rep_stride=8)

    with tik_instance.for_range(0, 1808) as idx:
        with tik_instance.for_range(0, 4) as i:
            box_dst[idx, i] = box_temp[idx, i]
    tik_instance.data_move(decode_l1[0, 0, 0], box_dst[0, 0], 0, 1, 16 * 1808 // 16, 0, 0)


def nms_result(tik_instance, gather_out_ub, proposals, all_cnt, valid_cnt):
    """
    gather nms result
    :param tik_instance:
    :param gather_out_ub: [100, 8]
    :param proposals: [100, 8]
    :param all_cnt: [0, 100]
    :param valid_cnt: [0, 100
    :return:
    """
    with tik_instance.for_range(0, valid_cnt) as i:
        with tik_instance.for_range(0, 8) as j:
            gather_out_ub[all_cnt + i, j].set_as(proposals[i, j])


def coord_process(tik_instance, box_encoding_gm, anchor_data, box_decode):
    """
    coord process
    :param tik_instance:
    :param box_encoding_gm: [1, 1808, 4]
    :param anchor_data: [1, 1808, 4]
    :param box_decode: [1, 4, 1808]
    :return:
    """
    # Decode gm (1, 1808, 4) -> l1 (1, 16, 1808) -> ub (1, 4, 1808)
    box_encoded_l1 = tik_instance.Tensor("float16", (1, 1808, 16), name="box_encoded_l1", scope=tik.scope_cbuf)  # 339k
    anchors_l1 = tik_instance.Tensor("float16", (1, 1808, 16), name="box_encoded_l1", scope=tik.scope_cbuf)  # 339k
    box_encoded_ub = tik_instance.Tensor("float16", (1, 4, 1808), name="box_encoded_ub", scope=tik.scope_ubuf)
    anchors_ub = tik_instance.Tensor("float16", (1, 4, 1808), name="anchors_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(box_encoded_l1[0, 0, 0], box_encoding_gm[0, 0, 0], 0, 1, 16 * 1808 // 16, 0, 0)
    with tik_instance.new_stmt_scope():
        pad_last_dim(tik_instance, anchor_data, anchors_l1)
    with tik_instance.new_stmt_scope():
        transpose_tensor(tik_instance, box_encoded_l1, box_encoded_ub)
    with tik_instance.new_stmt_scope():
        transpose_tensor(tik_instance, anchors_l1, anchors_ub)

    with tik_instance.new_stmt_scope():
        decode(tik_instance, box_encoded_ub.reshape((4, 1808)), anchors_ub.reshape((4, 1808)), [10.0, 10.0, 5.0, 5.0])
    tik_instance.data_move(box_decode[0, 0, 0], anchors_ub[0, 0, 0], 0, 1, 1808 * 4 // 16, 1, 1)


def cls_prob(tik_instance, scores_gm, scores_l1):
    """
    sigmoid scores and dimension convert
    :param tik_instance:
    :param scores_gm: [1, 1808, 96]
    :param scores_l1: [1, 96, 1808]
    :return:
    """
    scores_temp_l1 = tik_instance.Tensor("float16", (1, 1808, ceil_div_offline(NMS_CLASSES, 16) * 16), name="scores_ub",
                                         scope=tik.scope_cbuf)
    with tik_instance.new_stmt_scope():
        class_score(tik_instance, scores_gm, 1.0, scores_temp_l1)  # 1.5W cycle
    with tik_instance.new_stmt_scope():
        transpose_tensor(tik_instance, scores_temp_l1, scores_l1)  # 57 cycle


def _shape_check(_box_encode_dic, _scores_dic, _anchors_dic, _boxout_dic):
    if len(_box_encode_dic.get("shape")) != 3:
        print(_box_encode_dic.get("shape"))
        raise RuntimeError("input dims must equal 3")
    if len(_scores_dic.get("shape")) != 3:
        raise RuntimeError("input dims must equal 3")
    if len(_anchors_dic.get("shape")) != 2:
        raise RuntimeError("anchor dims must equal 2")
    if len(_boxout_dic.get("shape")) != 2:
        raise RuntimeError("output dims must equal 2")

    input_dtype = _box_encode_dic.get("dtype")
    if input_dtype != "float16":
        raise RuntimeError("input dtype only support float16!")

    input_h1 = _box_encode_dic.get("shape")[1]
    input_w1 = _box_encode_dic.get("shape")[2]
    if input_w1 != 16:
        raise RuntimeError("input coord last dim must align to 16!")

    input_h2 = _scores_dic.get("shape")[1]

    input_h3 = _anchors_dic.get("shape")[0]
    if input_h1 != input_h2 or input_h1 != input_h3:
        raise RuntimeError("input shape must equal!")

    output_w4 = _boxout_dic.get("shape")[1]
    if output_w4 != 8:
        raise RuntimeError("output coord last dim must equal 8!")


def yolov2_post_process(box_encode_dic, scores_dic, anchors_dic, boxout_dic, kernel_name="Yolov2PostProcess"):
    _shape_check(box_encode_dic, scores_dic, anchors_dic, boxout_dic)

    tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    crop_window = Window()
    g_buf = GBuf(tik_instance)

    # simoid scores
    with tik_instance.new_stmt_scope():
        cls_prob(tik_instance, g_buf.scores_gm, g_buf.scores_l1)

    # convert coordinates
    with tik_instance.new_stmt_scope():
        coord_process(tik_instance, g_buf.box_shadow_gm, g_buf.anchor_data, g_buf.decode_l1)

    # slice ub for each loop
    scores_slice = tik_instance.Tensor("float16", (1808, 1), name="scores_greater", scope=tik.scope_ubuf)
    ty1_ub = tik_instance.Tensor("float16", (1, 1808), name="ty1_ub", scope=tik.scope_ubuf)  # anchor y
    tx1_ub = tik_instance.Tensor("float16", (1, 1808), name="tx1_ub", scope=tik.scope_ubuf)
    ty2_ub = tik_instance.Tensor("float16", (1, 1808), name="ty2_ub", scope=tik.scope_ubuf)
    tx2_ub = tik_instance.Tensor("float16", (1, 1808), name="tx2_ub", scope=tik.scope_ubuf)  # (90,112)
    tik_instance.data_move(ty1_ub, g_buf.decode_l1[0, 0, :], 0, 1, 1808 // 16, 0, 0)
    tik_instance.data_move(tx1_ub, g_buf.decode_l1[0, 1, :], 0, 1, 1808 // 16, 0, 0)
    tik_instance.data_move(ty2_ub, g_buf.decode_l1[0, 2, :], 0, 1, 1808 // 16, 0, 0)
    tik_instance.data_move(tx2_ub, g_buf.decode_l1[0, 3, :], 0, 1, 1808 // 16, 0, 0)

    # ClipToWindow
    with tik_instance.new_stmt_scope():
        clip_to_window(tik_instance, ty1_ub, tx1_ub, ty2_ub, tx2_ub, 1808, crop_window)

    all_cnt = tik_instance.Scalar("uint64", name="all_cnt", init_value=0)
    valid_cnt = tik_instance.Scalar("uint64", name="valid_cnt")
    count = tik_instance.Scalar("uint16", name="count")
    with tik_instance.for_range(0, NMS_CLASSES) as loop_num:
        # build label
        tik_instance.vadds(MASK_128, g_buf.label_tensor_fp16[0, 0], g_buf.label_tensor_fp16[0, 0], 1.0, 1, 1, 1, 8, 8)

        # build proposal, filter out first row score
        tik_instance.data_move(scores_slice, g_buf.scores_l1[0, loop_num + 1, 0], 0, 1, 1808 // 16, 1, 1)
        with tik_instance.new_stmt_scope():
            filter_greater_than(tik_instance, SCORE_THRESHOLD, scores_slice, 1808, count)

        tik_instance.vconcat(g_buf.proposal_ub, ty1_ub, 1808 // 16, 0)  # y1,16 proposal each time
        tik_instance.vconcat(g_buf.proposal_ub, tx1_ub, 1808 // 16, 1)  # x1
        tik_instance.vconcat(g_buf.proposal_ub, ty2_ub, 1808 // 16, 2)  # y2
        tik_instance.vconcat(g_buf.proposal_ub, tx2_ub, 1808 // 16, 3)  # x2
        tik_instance.vconcat(g_buf.proposal_ub, scores_slice, 1808 // 16, 4)  # score

        # call NMS
        with tik_instance.new_stmt_scope():
            non_max_suppression(tik_instance, g_buf.proposal_ub, g_buf.proposals_out, count, valid_cnt)

        # pad or clip
        with tik_instance.if_scope(all_cnt < MAX_TOTAL_DETECTIONS):
            with tik_instance.if_scope((all_cnt + valid_cnt) >= MAX_TOTAL_DETECTIONS):
                valid_cnt.set_as(MAX_TOTAL_DETECTIONS - all_cnt)
        with tik_instance.else_scope():
            valid_cnt.set_as(0)

        # move box to gm
        with tik_instance.if_scope(valid_cnt > 0):
            # add label
            tik_instance.vadd([0x2020202020202020, 0x2020202020202020], g_buf.proposals_out[0, 0],
                              g_buf.proposals_out[0, 0], g_buf.label_tensor_fp16[0, 0], 8, 1, 1, 0, 8, 8, 0)

            with tik_instance.new_stmt_scope():
                nms_result(tik_instance, g_buf.gather_out_ub, g_buf.proposals_out, all_cnt, valid_cnt)
            all_cnt.set_as(all_cnt + valid_cnt)

    # move data from ub to gm
    tik_instance.data_move(g_buf.boxout_gm, g_buf.gather_out_ub, 0, 1, MAX_TOTAL_DETECTIONS * 8 // 16, 0, 0)

    tik_instance.BuildCCE(kernel_name, inputs=[g_buf.box_encoding_gm, g_buf.scores_gm, g_buf.anchor_data],
                          outputs=[g_buf.boxout_gm], enable_l2=False)
    return tik_instance
