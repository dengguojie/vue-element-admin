# -*- coding: utf-8 -*-
l1_fusion_type =  1
fm_addr_type =  1
out_addr_type =  1
out_16_addr_type = 1

l1_space = 0

dict1 = {
        "l1_fusion_type": l1_fusion_type,
        "l1_space": l1_space,

        "fm_addr_type": fm_addr_type,
        "out_addr_type": out_addr_type,
        "out_16_addr_type": out_16_addr_type,

        "fm_valid_shape": [],
        "fm_offset": [],
        "ws_s4_valid_shape":[],
        "ws_s8_valid_shape":[],
        "ws_fp16_valid_shape":[],
        "l1fusion_stride_swrite": 0, # 跳写的stride 不开启时为0
    }


rmpad_testcase = {
    "v200": {
        "st": (
            # 5 conv(int4) + dequant
            [5, (3, 24, 2, 58), (17, 24, 1, 1), (0, 0, 0, 0), (1, 1), 0, 0, 0, 1, 0, dict1],
            [5, (3, 24, 2, 58), (17, 24, 1, 1), (0, 0, 0, 0), (1, 1), 0, 0, 0, 0, 0, dict1],
            [5, (3, 24, 2, 64), (17, 24, 1, 1), (0, 0, 0, 0), (1, 1), 0, 0, 0, 1, 0, dict1],
            [5, (3, 24, 2, 64), (17, 24, 1, 1), (0, 0, 0, 0), (1, 1), 0, 0, 0, 0, 0, dict1]
            ),
        "bbit": ()
    },
}