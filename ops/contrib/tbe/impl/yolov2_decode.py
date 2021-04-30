# -*- coding: UTF-8 -*-
from te import tik


def calculate_loop(tik_instance, box_predictor, anchors, y_center_ub, x_center_ub, scale_factors, mask, offset,
                   repeat_times, num_div=0.5):
    # calculate anchors xcenter, ycenter, width, height
    tik_instance.vsub(mask, anchors[3, offset], anchors[3, offset], anchors[1, offset], repeat_times, 1, 1, 1, 8, 8, 8)
    tik_instance.vsub(mask, anchors[2, offset], anchors[2, offset], anchors[0, offset], repeat_times, 1, 1, 1, 8, 8, 8)
    tik_instance.vmuls(mask, y_center_ub, anchors[2, offset], num_div, repeat_times, 1, 1, 8, 8)
    tik_instance.vmuls(mask, x_center_ub, anchors[3, offset], num_div, repeat_times, 1, 1, 8, 8)
    tik_instance.vadd(mask, y_center_ub, y_center_ub, anchors[0, offset], repeat_times, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(mask, x_center_ub, x_center_ub, anchors[1, offset], repeat_times, 1, 1, 1, 8, 8, 8)

    # box_predictor div scale_factors
    tik_instance.vmuls(mask, box_predictor, box_predictor, 1.0 / scale_factors[0], repeat_times, 1, 1, 8, 8)
    tik_instance.vmuls(mask, box_predictor[1, offset], box_predictor[1, offset], 1.0 / scale_factors[1], repeat_times,
                       1, 1, 8, 8)
    tik_instance.vmuls(mask, box_predictor[2, offset], box_predictor[2, offset], 1.0 / scale_factors[2], repeat_times,
                       1, 1, 8, 8)
    tik_instance.vmuls(mask, box_predictor[3, offset], box_predictor[3, offset], 1.0 / scale_factors[3], repeat_times,
                       1, 1, 8, 8)

    # calculate w and h
    tik_instance.vexp(mask, box_predictor[2, offset], box_predictor[2, offset], repeat_times, 1, 1, 8, 8)
    tik_instance.vexp(mask, box_predictor[3, offset], box_predictor[3, offset], repeat_times, 1, 1, 8, 8)
    tik_instance.vmul(mask, box_predictor[2, offset], box_predictor[2, offset], anchors[2, offset], repeat_times, 1, 1,
                      1, 8, 8, 8)
    tik_instance.vmul(mask, box_predictor[3, offset], box_predictor[3, offset], anchors[3, offset], repeat_times, 1, 1,
                      1, 8, 8, 8)

    calculate_tail(tik_instance, box_predictor, anchors, y_center_ub, x_center_ub,
                   mask, offset, repeat_times, num_div)


def calculate_tail(tik_instance, box_predictor, anchors, y_center_ub, x_center_ub,
                   mask, offset, repeat_times, num_div=0.5):
    # calculate xcenter and ycenter
    tik_instance.vmul(mask, box_predictor, box_predictor, anchors[2, offset], repeat_times, 1, 1, 1, 8, 8, 8)
    tik_instance.vmul(mask, box_predictor[1, offset], box_predictor[1, offset], anchors[3, offset], repeat_times, 1, 1,
                      1, 8, 8, 8)
    tik_instance.vadd(mask, box_predictor, box_predictor, y_center_ub, repeat_times, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(mask, box_predictor[1, offset], box_predictor[1, offset], x_center_ub, repeat_times, 1, 1, 1, 8,
                      8, 8)

    # calculate ymin, xmin, ymax, xmax
    tik_instance.vmuls(mask, box_predictor[2, offset], box_predictor[2, offset], num_div, repeat_times, 1, 1, 8, 8)
    tik_instance.vmuls(mask, box_predictor[3, offset], box_predictor[3, offset], num_div, repeat_times, 1, 1, 8, 8)
    tik_instance.vsub(mask, anchors, box_predictor, box_predictor[2, offset], repeat_times, 1, 1, 1, 8, 8, 8)
    tik_instance.vsub(mask, anchors[1, offset], box_predictor[1, offset], box_predictor[3, offset], repeat_times, 1, 1,
                      1, 8, 8, 8)
    tik_instance.vadd(mask, anchors[2, offset], box_predictor, box_predictor[2, offset], repeat_times, 1, 1, 1, 8, 8, 8)
    tik_instance.vadd(mask, anchors[3, offset], box_predictor[1, offset], box_predictor[3, offset], repeat_times, 1, 1,
                      1, 8, 8, 8)


def decode(tik_instance, box_predictor, anchors, scale_factors):
    # get anchor num of hole grapgh
    num = box_predictor.shape[1]

    y_center_ub = tik_instance.Tensor("float16", (num,), name="y_center_ub", scope=tik.scope_ubuf)
    x_center_ub = tik_instance.Tensor("float16", (num,), name="x_center_ub", scope=tik.scope_ubuf)

    # judge how many times need to repeat, and the last data calculate once time
    repeat_times = num // 128
    data_last = num % 128
    data_last_index = num - data_last

    # when repeat_times > 0
    with tik_instance.if_scope(repeat_times > 0):
        calculate_loop(tik_instance, box_predictor, anchors, y_center_ub, x_center_ub, scale_factors, 128, 0,
                       repeat_times)

    # calculate the last data
    if data_last > 0:
        calculate_loop(tik_instance, box_predictor, anchors, y_center_ub, x_center_ub, scale_factors, data_last,
                       data_last_index, 1)
