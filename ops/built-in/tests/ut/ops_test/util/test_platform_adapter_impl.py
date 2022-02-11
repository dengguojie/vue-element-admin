#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# import tensorflow as tf
from op_test_frame.ut import OpUT
ut_case = OpUT("platform_adapter", "impl.util.platform_adapter")


# pylint: disable=unused-argument, import-outside-toplevel
def test_platform_api(test_arg):
    """
    test for platform api
    """
    from impl.util.platform_adapter import PlatformApi
    assert PlatformApi.api_check_support
    assert PlatformApi.scope_cbuf
    assert PlatformApi.scope_ubuf
    assert PlatformApi.scope_ca
    assert PlatformApi.scope_cb
    assert PlatformApi.scope_cc
    assert PlatformApi.scope_reg
    assert PlatformApi.scope_vreg
    assert PlatformApi.scope_preg
    assert PlatformApi.scope_areg
    assert PlatformApi.scope_ureg
    assert PlatformApi.scope_wreg
    assert PlatformApi.scope_aicpu
    assert PlatformApi.scope_gm
    assert PlatformApi.scope_cbuf_fusion
    assert PlatformApi.scope_smask
    assert PlatformApi.dma_copy
    assert PlatformApi.dma_copy_global
    assert PlatformApi.intrinsic_check_support
    assert PlatformApi.fusion_manager
    assert PlatformApi.cce_build


ut_case.add_cust_test_func("all", test_func=test_platform_api)


if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
