#!/usr/bin/env python3
# # -*- coding:utf-8 -*-
"""
test_iou_3d_golden
"""

'''
Special golden data generation function for ops add
'''
import numpy as np
import math


def get_area(c1, c2, c3):
    x1 = c1[0]
    y1 = c1[1]
    x2 = c2[0]
    y2 = c2[1]
    x3 = c3[0]
    y3 = c3[1]

    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    return area / 2


def calc_expect_func(boxes, query_boxes):
    batch = boxes.shape[0]
    n = boxes.shape[2]
    k = query_boxes.shape[2]

    res = np.zeros((batch, n, k))
    for b in range(batch):
        for i in range(n):
            rect1 = boxes[b, :, i]
            v1 = rect1[3] * rect1[4] * rect1[5]
            b1_min = rect1[2] - 0.5 * rect1[5]
            b1_max = rect1[2] + 0.5 * rect1[5]
            for j in range(k):
                rect2 = query_boxes[b, :, j]
                v2 = rect2[3] * rect2[4] * rect2[5]
                b2_min = rect2[2] - 0.5 * rect2[5]
                b2_max = rect2[2] + 0.5 * rect2[5]
                # step1 find vertex coordinates
                b1_sin = math.sin(rect1[6])
                b2_sin = math.sin(rect2[6])
                b1_cos = math.cos(rect1[6])
                b2_cos = math.cos(rect2[6])

                b1_x1 = rect1[0] - 0.5 * rect1[3] * b1_cos - 0.5 * rect1[4] * b1_sin
                b2_x1 = rect2[0] - 0.5 * rect2[3] * b2_cos - 0.5 * rect2[4] * b2_sin
                b1_y1 = rect1[1] - 0.5 * rect1[3] * b1_sin + 0.5 * rect1[4] * b1_cos
                b2_y1 = rect2[1] - 0.5 * rect2[3] * b2_sin + 0.5 * rect2[4] * b2_cos

                b1_x2 = rect1[0] + 0.5 * rect1[3] * b1_cos - 0.5 * rect1[4] * b1_sin
                b2_x2 = rect2[0] + 0.5 * rect2[3] * b2_cos - 0.5 * rect2[4] * b2_sin
                b1_y2 = rect1[1] + 0.5 * rect1[3] * b1_sin + 0.5 * rect1[4] * b1_cos
                b2_y2 = rect2[1] + 0.5 * rect2[3] * b2_sin + 0.5 * rect2[4] * b2_cos

                b1_x3 = rect1[0] + 0.5 * rect1[3] * b1_cos + 0.5 * rect1[4] * b1_sin
                b2_x3 = rect2[0] + 0.5 * rect2[3] * b2_cos + 0.5 * rect2[4] * b2_sin
                b1_y3 = rect1[1] + 0.5 * rect1[3] * b1_sin - 0.5 * rect1[4] * b1_cos
                b2_y3 = rect2[1] + 0.5 * rect2[3] * b2_sin - 0.5 * rect2[4] * b2_cos

                b1_x4 = rect1[0] - 0.5 * rect1[3] * b1_cos + 0.5 * rect1[4] * b1_sin
                b2_x4 = rect2[0] - 0.5 * rect2[3] * b2_cos + 0.5 * rect2[4] * b2_sin
                b1_y4 = rect1[1] - 0.5 * rect1[3] * b1_sin - 0.5 * rect1[4] * b1_cos
                b2_y4 = rect2[1] - 0.5 * rect2[3] * b2_sin - 0.5 * rect2[4] * b2_cos

                b1_x = [b1_x1, b1_x2, b1_x3, b1_x4]
                b1_y = [b1_y1, b1_y2, b1_y3, b1_y4]
                b2_x = [b2_x1, b2_x2, b2_x3, b2_x4]
                b2_y = [b2_y1, b2_y2, b2_y3, b2_y4]

                # step2 find the coordinates of the intersection
                corners = []
                AB_x = b2_x2 - b2_x1
                AB_y = b2_y2 - b2_y1
                AD_x = b2_x4 - b2_x1
                AD_y = b2_y4 - b2_y1

                AB_AB = AB_x * AB_x + AB_y * AB_y
                AD_AD = AD_x * AD_x + AD_y * AD_y
                for index in range(4):
                    b1_x_ = b1_x[index]
                    b1_y_ = b1_y[index]
                    AP_x = b1_x_ - b2_x1
                    AP_y = b1_y_ - b2_y1

                    AB_AP = AB_x * AP_x + AB_y * AP_y
                    AD_AP = AD_x * AP_x + AD_y * AP_y
                    if 0 <= AB_AP <= AB_AB:
                        if 0 <= AD_AP <= AD_AD:
                            corners.append([b1_x_, b1_y_])

                AB_x = b1_x2 - b1_x1
                AB_y = b1_y2 - b1_y1
                AD_x = b1_x4 - b1_x1
                AD_y = b1_y4 - b1_y1

                AB_AB = AB_x * AB_x + AB_y * AB_y
                AD_AD = AD_x * AD_x + AD_y * AD_y
                for index in range(4):
                    b2_x_ = b2_x[index]
                    b2_y_ = b2_y[index]
                    AP_x = b2_x_ - b1_x1
                    AP_y = b2_y_ - b1_y1

                    AB_AP = AB_x * AP_x + AB_y * AP_y
                    AD_AP = AD_x * AP_x + AD_y * AP_y
                    if 0 <= AB_AP <= AB_AB:
                        if 0 <= AD_AP <= AD_AD:
                            corners.append([b2_x_, b2_y_])

                for b1_i in range(4):
                    # A
                    b1_x1 = b1_x[b1_i]
                    b1_y1 = b1_y[b1_i]
                    # B
                    b1_x2 = b1_x[(b1_i + 1) % 4]
                    b1_y2 = b1_y[(b1_i + 1) % 4]

                    for b2_i in range(4):
                        # C
                        b2_x1 = b2_x[b2_i]
                        b2_y1 = b2_y[b2_i]
                        # D
                        b2_x2 = b2_x[(b2_i + 1) % 4]
                        b2_y2 = b2_y[(b2_i + 1) % 4]

                        AC_x = b2_x1 - b1_x1
                        AC_y = b2_y1 - b1_y1
                        AD_x = b2_x2 - b1_x1
                        AD_y = b2_y2 - b1_y1
                        BC_x = b2_x1 - b1_x2
                        BC_y = b2_y1 - b1_y2
                        BD_x = b2_x2 - b1_x2
                        BD_y = b2_y2 - b1_y2

                        ACxAD = AC_x * AD_y - AD_x * AC_y
                        BCxBD = BC_x * BD_y - BD_x * BC_y
                        sign1 = ACxAD * BCxBD
                        if (sign1 < 0):
                            CAxCB = AC_x * BC_y - BC_x * AC_y
                            DAxDB = AD_x * BD_y - BD_x * AD_y
                            sign2 = CAxCB * DAxDB
                            if (sign2 < 0):
                                fenmu = (b1_x1 - b1_x2) * (b2_y1 - b2_y2) - (b1_y1 - b1_y2) * (b2_x1 - b2_x2)
                                tmp_1 = (b1_x1 * b1_y2 - b1_y1 * b1_x2)
                                tmp_2 = (b2_x1 * b2_y2 - b2_y1 * b2_x2)
                                px = (tmp_1 * (b2_x1 - b2_x2) - (b1_x1 - b1_x2) * tmp_2) / fenmu
                                py = (tmp_1 * (b2_y1 - b2_y2) - (b1_y1 - b1_y2) * tmp_2) / fenmu

                                corners.append([px, py])

                max_of_min = max(b2_min, b1_min)
                min_of_max = min(b2_max, b1_max)
                real_d = max(min_of_max - max_of_min, 0)
                # step2 find the overlapping area
                corner_nums = len(corners)
                corners = np.array(corners)
                area = 0
                if corner_nums == 3:
                    area = get_area(corners[0], corners[1], corners[2])
                if corner_nums > 3:
                    x = corners[:, 0]
                    y = corners[:, 1]
                    center_spot = [sum(x) / corner_nums, sum(y) / corner_nums]

                    count = [-1] * 16
                    left = 0
                    right = 0
                    x_x1 = x - center_spot[0]
                    y_y1 = y - center_spot[1]
                    box_xie = y_y1 / x_x1
                    idx_tensor = np.argsort(box_xie)

                    for idx in range(corner_nums):
                        tmp_spot = corners[idx_tensor[idx]]
                        if tmp_spot[0] > center_spot[0]:
                            count[left] = idx_tensor[idx]
                            left += 1
                        else:
                            count[8 + right] = idx_tensor[idx]
                            right += 1

                    old_re = count[0]
                    for idx in range(1, 15):
                        if count[idx] != -1:
                            area += get_area(center_spot, corners[old_re], corners[count[idx]])
                            old_re = count[idx]
                    area += get_area(center_spot, corners[count[0]], corners[old_re])
                inter_volume = area * real_d
                iou = inter_volume / (v1 + v2 - inter_volume + 1e-6)
                res[b, i, j] = iou
    return res
