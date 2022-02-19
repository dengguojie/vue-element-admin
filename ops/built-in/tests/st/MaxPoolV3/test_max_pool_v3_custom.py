#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from importlib import reload

# pylint: disable=import-outside-toplevel
def test_get_op_support_info_nd():
    """
    test for get_op_support_info
    """
    import sys
    import json
    from impl.max_pool_v3 import get_op_support_info
    reload(sys.modules.get("impl.max_pool_v3"))
    support_info = get_op_support_info(
        {"dtype": "float16", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 64, 56, 56),
         "shape": (1, 64, 56, 56)},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 64, 28, 28),
         "shape": (1, 4, 28, 28, 16)},
        [1, 1, 3, 3], [1, 1, 2, 2], "SAME", (1, 1, 1, 1), "NCHW", False, False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 0


def test_get_op_support_info_same():
    """
    test for get_op_support_info
    """
    import sys
    import json
    from impl.max_pool_v3 import get_op_support_info
    reload(sys.modules.get("impl.max_pool_v3"))
    support_info = get_op_support_info(
        {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1, 64, 56, 56),
         "shape": (1, 4, 56, 56, 16)},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 64, 28, 28),
         "shape": (1, 4, 28, 28, 16)},
        [1, 1, 3, 3], [1, 1, 2, 2], "SAME", (1, 1, 1, 1), "NC1HWC0", False, False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 1
    for item in split_maps:
        input_list = item.get("inputList")
        assert len(input_list) == 1
        axis = input_list[0].get("axis")
        assert len(axis) == 1
        assert axis[0] in (0,)


if __name__ == '__main__':
    test_get_op_support_info_nd()
    test_get_op_support_info_same()
