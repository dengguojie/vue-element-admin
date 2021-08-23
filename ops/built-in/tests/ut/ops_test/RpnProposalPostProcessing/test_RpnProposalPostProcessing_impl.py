#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("RpnProposalPostProcessing", None, None)

# ============ add cases here ===============

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])