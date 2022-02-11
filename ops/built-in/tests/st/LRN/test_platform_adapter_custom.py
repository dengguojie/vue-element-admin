#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from importlib import reload

# pylint: disable=import-outside-toplevel
def test_platform_api():
    """
    test for platform api
    """
    from impl.util.platform_adapter import PlatformApi
    import sys
    reload(sys.modules.get("impl.util.platform_adapter"))
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


if __name__ == "__main__":
    test_platform_api()
