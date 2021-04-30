# -*- coding: UTF-8 -*-
from te import tik


# @normalize to 0~1
# @score_data_gma as input tensor，outdata as output tensor.

def class_score(tik_instance, score_data_gm, scale, outdata):
    """

    :param tik_instance:
    :param score_data_gm: [1, 96, 1808]
    :param scale: 1.0
    :param outdata: [1, 96, 1808]
    :return:
    """
    # judge how many times to move data, set at most 238K
    dim1 = score_data_gm.shape[1]
    dim2 = score_data_gm.shape[2]
    dim1_num = 200 * 512 // dim2  # 5
    cal_times = dim1 * dim2 // (200 * 512)  # 1
    score_data = tik_instance.Tensor("float16", (1, dim1_num, dim2), name="score_data", scope=tik.scope_ubuf)
    div_scale = -1.0 / scale
    all_one = tik_instance.Tensor("float16", (16,), name="all_one_ub", scope=tik.scope_ubuf)
    tik_instance.vector_dup(16, all_one, 1.0, 1, 1, 8)

    # when all data size big than 240KB, need to slice
    for i in range(0, cal_times + 1):
        if i < cal_times:
            repeat_times = dim1_num * dim2 // 128
            data_last = dim1_num * dim2 % 128
            data_last_index = dim1_num * dim2 - data_last
            tik_instance.data_move(score_data[0, 0, 0], score_data_gm[0, i * dim1_num, 0], 0, 1, dim1_num * dim2 // 16,
                                   0, 0)
        else:
            repeat_times = (dim1 - i * dim1_num) * dim2 // 128
            data_last = (dim1 - i * dim1_num) * dim2 % 128
            data_last_index = (dim1 - i * dim1_num) * dim2 - data_last
            tik_instance.data_move(score_data[0, 0, 0], score_data_gm[0, i * dim1_num, 0], 0, 1,
                                   (dim1 - i * dim1_num) * dim2 // 16, 0, 0)

        with tik_instance.if_scope(repeat_times > 255):
            cnt = repeat_times // 255
            with tik_instance.for_range(0, cnt) as tmp:
                len1 = tmp * 255 * 128
                tik_instance.vmuls(128, score_data[len1], score_data[len1], div_scale, 255, 1, 1, 8, 8)
                tik_instance.vexp(128, score_data[len1], score_data[len1], 255, 1, 1, 8, 8)  # e(-x)
                tik_instance.vadd(128, score_data[len1], score_data[len1], all_one, 255, 1, 1, 0, 8, 8, 0)
                tik_instance.vrec(128, score_data[len1], score_data[len1], 255, 1, 1, 8, 8)

        repeat_num = repeat_times % 255
        # scale logits
        with tik_instance.if_scope(repeat_num > 0):
            len2 = cnt * 255 * 128
            tik_instance.vmuls(128, score_data[len2], score_data[len2], div_scale, repeat_num, 1, 1, 8, 8)
            tik_instance.vexp(128, score_data[len2], score_data[len2], repeat_num, 1, 1, 8, 8)  # e(-x)
            tik_instance.vadd(128, score_data[len2], score_data[len2], all_one, repeat_num, 1, 1, 0, 8, 8, 0)
            tik_instance.vrec(128, score_data[len2], score_data[len2], repeat_num, 1, 1, 8, 8)

        with tik_instance.if_scope(data_last != 0):
            tik_instance.vmuls(data_last, score_data[data_last_index], score_data[data_last_index], div_scale, 1, 1, 1,
                               8, 8)
            tik_instance.vexp(data_last, score_data[data_last_index], score_data[data_last_index], 1, 1, 1, 8, 8)
            tik_instance.vadd(data_last, score_data[data_last_index], score_data[data_last_index], all_one, 1, 1, 1, 0,
                              8, 8, 0)
            tik_instance.vrec(data_last, score_data[data_last_index], score_data[data_last_index], 1, 1, 1, 8, 8)

        with tik_instance.if_scope(i < cal_times):
            tik_instance.data_move(outdata[0, i * dim1_num, 0], score_data, 0, 1, dim1_num * dim2 // 16, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(outdata[0, i * dim1_num, 0], score_data, 0, 1, (dim1 - dim1_num) * dim2 // 16, 0, 0)
