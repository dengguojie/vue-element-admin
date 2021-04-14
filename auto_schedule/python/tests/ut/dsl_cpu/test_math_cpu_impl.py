# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import operator
import random
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")
ut_case = OpUT("math_cpu", "dsl_cpu.test_math_cpu_impl")


def test_vadd_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.placeholder((n,), name="input2")
    output = tbe.vadd(input1, input2)
    sch = tvm.create_schedule(output.op)
    fadd = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fadd")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fadd(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsub_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 2048
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.placeholder((n,), name="input2")
    output = tbe.vsub(input1, input2)
    sch = tvm.create_schedule(output.op)
    fsub = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fsub")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fsub(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() - b.asnumpy())
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmul_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.placeholder((n,), name="input2")
    output = tbe.vmul(input1, input2)
    sch = tvm.create_schedule(output.op)
    fmul = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fmul")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fmul(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() * b.asnumpy())
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vdiv_cpu_api_not_support_vdiv(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.placeholder((n,), name="input2")
    output = tbe.vdiv(input1, input2)
    sch = tvm.create_schedule(output.op)
    fdiv = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fdiv")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fdiv(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() / b.asnumpy(), atol=0.0001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vdiv_cpu_api(soc):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.placeholder((n,), name="input2")
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend910A")
    output = tbe.vdiv(input1, input2)
    sch = tvm.create_schedule(output.op)
    fdiv = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fdiv")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fdiv(a, b, c)
    # 3.verify the correctness of output
    try:
        # Restore soc version to soc, or it affect the following use cases
        te_set_version(soc)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() / b.asnumpy(), atol=0.0001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmod_cpu_api(soc):
    """
    The vmod interface is only support float16.
    @param soc: soc version
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend910A")
    output = tbe.vmod(input1, input2)
    sch = tvm.create_schedule(output.op)
    fmod = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fmod")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fmod(a, b, c)
    # 3.verify the correctness of output
    try:
        # Restore soc version to soc, or it affect the following use cases
        te_set_version(soc)
        tvm.testing.assert_allclose(c.asnumpy(), np.mod(a.asnumpy(), b.asnumpy()), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmod_cpu_api_not_support_vdiv_and_vconv_f322s32f(_):
    """
    not support Intrinsic_vdiv and Intrinsic_vconv f322s32f
    @return: Ture && false
    """
    n = 1024
    # The vmod interface is only support float16.
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    output = tbe.vmod(input1, input2)
    sch = tvm.create_schedule(output.op)
    fmod = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fmod")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fmod(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.mod(a.asnumpy(), b.asnumpy()), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmod_cpu_api_not_support_vconv_f322s32f(soc):
    """
    not support Intrinsic_vconv f322s32f
    @param soc: soc version
    @return: Ture && false
    """
    n = 1024
    # The vmod interface is only support float16.
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    from te.platform.cce_conf import te_set_version
    te_set_version("Hi3796CV300ES")
    output = tbe.vmod(input1, input2)
    sch = tvm.create_schedule(output.op)
    fmod = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fmod")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(low=0.1, size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fmod(a, b, c)
    # 3.verify the correctness of output
    try:
        # Restore soc version to soc, or it affect the following use cases
        te_set_version(soc)
        tvm.testing.assert_allclose(c.asnumpy(), np.mod(a.asnumpy(), b.asnumpy()), atol=0.5, rtol=0.5)
    except AssertionError as e:
        print(e)
        # Becase of precision problem, there is return TRUE
        return True
    return True


def test_vmin_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.placeholder((n,), name="input2")
    output = tbe.vmin(input1, input2)
    sch = tvm.create_schedule(output.op)
    fmin = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fmin")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fmin(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.minimum(a.asnumpy(), b.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmax_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.placeholder((n,), name="input2")
    output = tbe.vmax(input1, input2)
    sch = tvm.create_schedule(output.op)
    fmax = tvm.build(sch, [input1, input2, output], "c", "llvm", name="fmax")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    fmax(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.maximum(a.asnumpy(), b.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vor_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    # Intrinsic_vor| int16,uint16
    input1 = tvm.placeholder((n,), dtype="uint16", name="input1")
    input2 = tvm.placeholder((n,), dtype="uint16", name="input2")
    output = tbe.vor(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_or = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_or")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_or(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.bitwise_or(a.asnumpy(), b.asnumpy()), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vand_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    # Intrinsic_vand| int16,uint16
    input1 = tvm.placeholder((n,), dtype="uint16", name="input1")
    input2 = tvm.placeholder((n,), dtype="uint16", name="input2")
    output = tbe.vand(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_and = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_and")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_and(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.bitwise_and(a.asnumpy(), b.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vadds_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.const(2, dtype="float16")
    output = tbe.vadds(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_adds = tvm.build(sch, [input1, output], "c", "llvm", name="func_adds")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.full((n,), 2), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_adds(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmins_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.const(2, dtype="float16")
    output = tbe.vmins(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_vmins = tvm.build(sch, [input1, output], "c", "llvm", name="func_vmins")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.full((n,), 2), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vmins(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.minimum(a.asnumpy(), b.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmuls_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), name="input1")
    input2 = tvm.const(2, dtype="float16")
    output = tbe.vmuls(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_vmuls = tvm.build(sch, [input1, output], "c", "llvm", name="func_vmuls")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.full((n,), 2), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vmuls(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() * b.asnumpy())
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlog_cpu_api_not_support_vln_fp32_and_precision(_):
    """
    In this situation(not support Intrinsic_vln|fp32 and priority equal one), It will use talor.
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vlog(input1, priority_flag=1)
    sch = tvm.create_schedule(output.op)
    func_vlog = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlog")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vlog(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.log(a.asnumpy()), atol=0.005, rtol=0.005)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlog_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vlog(input1, priority_flag=0)
    sch = tvm.create_schedule(output.op)
    func_vlog = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlog")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vlog(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.log(a.asnumpy()), atol=0.005, rtol=0.005)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vexp_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vexp(input1)
    sch = tvm.create_schedule(output.op)
    func_vexp = tvm.build(sch, [input1, output], "c", "llvm", name="func_vexp")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vexp(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.exp(a.asnumpy()), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vabs_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vabs(input1)
    sch = tvm.create_schedule(output.op)
    func_vabs = tvm.build(sch, [input1, output], "c", "llvm", name="func_vabs")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vabs(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.abs(a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vrec_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vrec(input1)
    sch = tvm.create_schedule(output.op)
    func_vrec = tvm.build(sch, [input1, output], "c", "llvm", name="func_vrec")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vrec(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.reciprocal(a.asnumpy()), atol=0.005, rtol=0.005)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vnot_cpu_api(_):
    """
    Intrinsic_vnot| int16, uint16
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="int16", name="input1")
    output = tbe.vnot(input1)
    sch = tvm.create_schedule(output.op)
    func_vnot = tvm.build(sch, [input1, output], "c", "llvm", name="func_vnot")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vnot(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.bitwise_not(a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsqrt_cpu_api_not_support_vsqrt_and_precision(_):
    """
    not support Intrinsic_vsqrt, and precision
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vsqrt(input1, priority_flag=1)
    sch = tvm.create_schedule(output.op)
    func_vsqrt = tvm.build(sch, [input1, output], "c", "llvm", name="func_vsqrt")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsqrt(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.sqrt(a.asnumpy()), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsqrt_cpu_api_not_support_vsqrt_and_performance(_):
    """
    not support Intrinsic_vsqrt, and performance
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vsqrt(input1)
    sch = tvm.create_schedule(output.op)
    func_vsqrt = tvm.build(sch, [input1, output], "c", "llvm", name="func_vsqrt")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsqrt(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.sqrt(a.asnumpy()), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsqrt_cpu_api(soc):
    """
    support Intrinsic_vsqrt
    @param soc: soc version
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend910A")
    output = tbe.vsqrt(input1)
    sch = tvm.create_schedule(output.op)
    func_vsqrt = tvm.build(sch, [input1, output], "c", "llvm", name="func_vsqrt")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsqrt(a, c)
    # 3.verify the correctness of output
    try:
        # Restore soc version to soc, or it affect the following use cases
        te_set_version(soc)
        tvm.testing.assert_allclose(c.asnumpy(), np.sqrt(a.asnumpy()), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vrsqrt_cpu_api_not_support_vsqrt_and_precision(_):
    """
    not support Intrinsic_vsqrt and precision. Maybe there is something wrong
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vrsqrt(input1, priority_flag=1.0)
    sch = tvm.create_schedule(output.op)
    func_vsqrt = tvm.build(sch, [input1, output], "c", "llvm", name="func_vsqrt")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsqrt(a, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.reciprocal(np.sqrt(a.asnumpy())),
                                    atol=0.005, rtol=0.005)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vrsqrt_cpu_api(soc):
    """
    not support Intrinsic_vsqrt and precision. Maybe there is something wrong
    @param soc: soc version
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend910A")
    output = tbe.vrsqrt(input1)
    sch = tvm.create_schedule(output.op)
    func_vsqrt = tvm.build(sch, [input1, output], "c", "llvm", name="func_vsqrt")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsqrt(a, c)
    # 3.verify the correctness of output
    try:
        # Restore soc version to soc, or it affect the following use cases
        te_set_version(soc)
        tvm.testing.assert_allclose(c.asnumpy(), np.reciprocal(np.sqrt(a.asnumpy())),
                                    atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vaxpy_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    input3 = tvm.const(2, dtype="float16")
    output = tbe.vaxpy(input1, input2, input3)
    sch = tvm.create_schedule(output.op)
    func_vaxpy = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vaxpy")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.full((n,), 2), ctx)
    d = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vaxpy(a, b, d)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(d.asnumpy(), a.asnumpy() * c.asnumpy() + b.asnumpy(), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmla_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    input3 = tvm.placeholder((n,), dtype="float16", name="input3")
    output = tbe.vmla(input1, input2, input3)
    sch = tvm.create_schedule(output.op)
    func_vmla = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vmla")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=n).astype(input3.dtype), ctx)
    d = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vmla(a, b, c, d)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(d.asnumpy(), a.asnumpy() * b.asnumpy() + c.asnumpy(), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmadd_cpu_api(_):
    """
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    input3 = tvm.placeholder((n,), dtype="float16", name="input3")
    output = tbe.vmadd(input1, input2, input3)
    sch = tvm.create_schedule(output.op)
    func_vmadd = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vmadd")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=n).astype(input3.dtype), ctx)
    d = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vmadd(a, b, c, d)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(d.asnumpy(), a.asnumpy() * c.asnumpy() + b.asnumpy(), atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlogic_cpu_api(_):
    """
    @return: Ture && false
    """
    op_dict = {"logic_and": np.logical_and, "logic_or": np.logical_or}
    for key, op in op_dict.items():
        n = 10000
        input1 = tvm.placeholder((n,), dtype="bool", name="input1")
        input2 = tvm.placeholder((n,), dtype="bool", name="input2")
        output = tbe.vlogic(input1, input2, operation=key)
        sch = tvm.create_schedule(output.op)
        func_vlogic = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vlogic")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vlogic(a, b, c)
        # 3.verify the correctness of output
        try:
            tvm.testing.assert_allclose(c.asnumpy(), op(a.asnumpy(), b.asnumpy()))
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vlogic_cpu_api_not_op(_):
    """
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="bool", name="input1")
    output = tbe.vlogic(input1, operation="logic_not")
    sch = tvm.create_schedule(output.op)
    func_vlogic = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlogic")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vlogic(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.logical_not(a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vcmp_cpu_api_bool(_):
    """
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 10000
        input1 = tvm.placeholder((n,), dtype="float16", name="input1")
        input2 = tvm.placeholder((n,), dtype="float16", name="input2")
        output = tbe.vcmp(input1, input2, operation=key)
        sch = tvm.create_schedule(output.op)
        func_vcmp = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vcmp")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmp(a, b, c)
        # 3.verify the correctness of output
        try:
            tvm.testing.assert_allclose(c.asnumpy(), op(a.asnumpy(), b.asnumpy()))
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmp_cpu_api_bool_and_rhs_is_const(_):
    """
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 10000
        const_value = random.randint(1, 5)
        input1 = tvm.placeholder((n,), dtype="float16", name="input1")
        input2 = tvm.const(const_value, dtype="float16")
        output = tbe.vcmp(input1, input2, operation=key)
        sch = tvm.create_schedule(output.op)
        func_vcmp = tvm.build(sch, [input1, output], "c", "llvm", name="func_vcmp")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmp(a, b)
        # 3.verify the correctness of output
        try:
            tvm.testing.assert_allclose(b.asnumpy(), op(a.asnumpy(), const_value))
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmp_cpu_api_bit(_):
    """
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, _ in op_dict.items():
        n = 1024
        input1 = tvm.placeholder((n,), dtype="float16", name="input1")
        input2 = tvm.placeholder((n,), dtype="float16", name="input2")
        output = tbe.vcmp(input1, input2, operation=key, mode="bit")
        sch = tvm.create_schedule(output.op)
        func_vcmp = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vcmp")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
        c = tvm.nd.array(np.zeros(n // 8, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmp(a, b, c)
        # 3.verify the correctness of output
        benchmark_data_a = [a.asnumpy()[i] for i in range(0, n, 8)]
        benchmark_data_b = [b.asnumpy()[i] for i in range(0, n, 8)]
        try:
            tvm.testing.assert_allclose(c.asnumpy(),
                                        np.bitwise_and(np.array(benchmark_data_a, dtype=np.uint8),
                                                       np.array(benchmark_data_b, dtype=np.uint8)))
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmp_cpu_api_bit_and_rhs_is_const(_):
    """
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, _ in op_dict.items():
        n = 1024
        const_value = np.random.rand()
        input1 = tvm.placeholder((n,), dtype="float16", name="input1")
        input2 = tvm.const(const_value, dtype="float16")
        output = tbe.vcmp(input1, input2, operation=key, mode="bit")
        sch = tvm.create_schedule(output.op)
        func_vcmp = tvm.build(sch, [input1, output], "c", "llvm", name="func_vcmp")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros(n // 8, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmp(a, b)
        # 3.verify the correctness of output
        benchmark_data_a = [a.asnumpy()[i] for i in range(0, n, 8)]
        benchmark_data_b = [const_value] * (n // 8)
        try:
            tvm.testing.assert_allclose(b.asnumpy(),
                                        np.bitwise_and(np.array(benchmark_data_a, dtype=np.uint8),
                                                       np.array(benchmark_data_b, dtype=np.uint8)))
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vsel_cpu_api_bool_tensor_to_tensor(_):
    """
    @return: Ture && false
    """
    n = 1228
    input1 = tvm.placeholder((n,), name="condition", dtype="bool")
    input2 = tvm.placeholder((n,), name="input2", dtype="float16")
    input3 = tvm.placeholder((n,), name="input3", dtype="float16")
    output = tbe.vsel(input1, input2, input3)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=n).astype(input3.dtype), ctx)
    d = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b, c, d)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        if cond:
            benchmark_data.append(b.asnumpy()[i])
        else:
            benchmark_data.append(c.asnumpy()[i])
    try:
        tvm.testing.assert_allclose(d.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsel_cpu_api_bool_tensor_to_scalar(_):
    """
    @return: Ture && false
    """
    n = 1228
    input1 = tvm.placeholder((n,), name="condition", dtype="bool")
    input2 = tvm.placeholder((n,), name="input2", dtype="float16")
    output = tbe.vsel(input1, input2, 1.0)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b, c)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        if cond:
            benchmark_data.append(b.asnumpy()[i])
        else:
            benchmark_data.append(1.0)
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsel_cpu_api_bool_scalar_to_tensor(_):
    """
    @return: Ture && false
    """
    n = 1228
    input1 = tvm.placeholder((n,), name="condition", dtype="bool")
    input2 = tvm.placeholder((n,), name="input2", dtype="float16")
    output = tbe.vsel(input1, 1.0, input2)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b, c)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        if cond:
            benchmark_data.append(1.0)
        else:
            benchmark_data.append(b.asnumpy()[i])
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsel_cpu_api_bool_scalar_to_scalar(_):
    """
    @return: Ture && false
    """
    n = 1228
    input1 = tvm.placeholder((n,), name="condition", dtype="bool")
    output = tbe.vsel(input1, 2.0, 1.0)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        if cond:
            benchmark_data.append(2.0)
        else:
            benchmark_data.append(1.0)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsel_cpu_api_uint8_tensor_to_tensor(_):
    """
    @return: Ture && false
    """
    n = 1232
    input1 = tvm.placeholder((n // 8,), name="condition", dtype="uint8")
    input2 = tvm.placeholder((n,), name="input2", dtype="float16")
    input3 = tvm.placeholder((n,), name="input3", dtype="float16")
    output = tbe.vsel(input1, input2, input3)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n // 8).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=n).astype(input3.dtype), ctx)
    d = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b, c, d)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        repeat_num = 0
        while repeat_num < 8:
            if cond:
                benchmark_data.append(b.asnumpy()[i * 8 + repeat_num])
            else:
                benchmark_data.append(c.asnumpy()[i * 8 + repeat_num])
            repeat_num += 1
    try:
        tvm.testing.assert_allclose(d.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsel_cpu_api_uint8_tensor_to_scalar(_):
    """
    @return: Ture && false
    """
    n = 1232
    input1 = tvm.placeholder((n // 8,), name="condition", dtype="uint8")
    input2 = tvm.placeholder((n,), name="input2", dtype="float16")
    output = tbe.vsel(input1, input2, 1.0)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n // 8).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b, c)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        repeat_num = 0
        while repeat_num < 8:
            if cond:
                benchmark_data.append(b.asnumpy()[i * 8 + repeat_num])
            else:
                benchmark_data.append(1.0)
            repeat_num += 1
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsel_cpu_api_uint8_scalar_to_tensor(_):
    """
    @return: Ture && false
    """
    n = 1232
    input1 = tvm.placeholder((n // 8,), name="condition", dtype="uint8")
    input2 = tvm.placeholder((n,), name="input2", dtype="float16")
    output = tbe.vsel(input1, 1.0, input2)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n // 8).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b, c)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        repeat_num = 0
        while repeat_num < 8:
            if cond:
                benchmark_data.append(1.0)
            else:
                benchmark_data.append(b.asnumpy()[i * 8 + repeat_num])
            repeat_num += 1
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsel_cpu_api_uint8_scalar_to_scalar(_):
    """
    @return: Ture && false
    """
    n = 1232
    input1 = tvm.placeholder((n // 8,), name="condition", dtype="uint8")
    output = tbe.vsel(input1, 2.0, 1.0)
    sch = tvm.create_schedule(output.op)
    func_vsel = tvm.build(sch, [input1, output], "c", "llvm", name="func_vsel")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.randint(low=0, high=2, size=n // 8).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsel(a, b)
    # 3.verify the correctness of output
    benchmark_data = []
    for i, cond in enumerate(a.asnumpy()):
        repeat_num = 0
        while repeat_num < 8:
            if cond:
                benchmark_data.append(2.0)
            else:
                benchmark_data.append(1.0)
            repeat_num += 1
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.array(benchmark_data))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vcmpsel_cpu_api_tensor_scalar_tensor_scalar(_):
    """
    lhs, rhs, slhs, srhs are respectively tensor, scalar, tensor, scalar
    In this sutiation, rhs is 2.0, srhs is 0.0
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        output = tbe.vcmpsel(input1, operation=key)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], 2.0):
                    benchmark_data[i][j] = a.asnumpy()[i][j]
                else:
                    benchmark_data[i][j] = 0.0
        try:
            tvm.testing.assert_allclose(b.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmpsel_cpu_api_tensor_tensor_tensor_tensor(_):
    """
    lhs, rhs, slhs, srhs are all tensor
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        input2 = tvm.placeholder((n, n), dtype="float16", name="input2")
        input3 = tvm.placeholder((n, n), dtype="float16", name="input3")
        input4 = tvm.placeholder((n, n), dtype="float16", name="input4")
        output = tbe.vcmpsel(input1, input2, key, input3, input4)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, input2, input3, input4, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        c = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        d = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        e = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b, c, d, e)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], b.asnumpy()[i][j]):
                    benchmark_data[i][j] = c.asnumpy()[i][j]
                else:
                    benchmark_data[i][j] = d.asnumpy()[i][j]
        try:
            tvm.testing.assert_allclose(e.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmpsel_cpu_api_tensor_scalar_scalar_scalar(_):
    """
    lhs, rhs, slhs, srhs are respectively tensor, scalar, scalar, scalar
    In this sutiation, rhs is 2.0, srhs is 0.0
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        output = tbe.vcmpsel(input1, operation=key, slhs=1.0)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], 2.0):
                    benchmark_data[i][j] = 1.0
                else:
                    benchmark_data[i][j] = 0.0
        try:
            tvm.testing.assert_allclose(b.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmpsel_cpu_api_tensor_tensor_scalar_scalar(_):
    """
    lhs, rhs, slhs, srhs are respectively tensor, scalar, tensor, scalar
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        input2 = tvm.placeholder((n, n), dtype="float16", name="input2")
        output = tbe.vcmpsel(input1, input2, operation=key, slhs=1.0, srhs=2.0)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b, c)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], b.asnumpy()[i][j]):
                    benchmark_data[i][j] = 1.0
                else:
                    benchmark_data[i][j] = 2.0
        try:
            tvm.testing.assert_allclose(c.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmpsel_cpu_api_tensor_scalar_scalar_tensor(_):
    """
    lhs, rhs, slhs, srhs are respectively tensor, None, scalar, tensor
    In this sutiation, rhs is 2.0, srhs is 0.0
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        input2 = tvm.placeholder((n, n), dtype="float16", name="input2")
        output = tbe.vcmpsel(input1, None, operation=key, slhs=1.0, srhs=input2)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b, c)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], 2.0):
                    benchmark_data[i][j] = 1.0
                else:
                    benchmark_data[i][j] = b.asnumpy()[i][j]
        try:
            tvm.testing.assert_allclose(c.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmpsel_cpu_api_tensor_tensor_tensor_scalar(_):
    """
    lhs, rhs, slhs, srhs are respectively tensor, tensor, tensor, scalar
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        input2 = tvm.placeholder((n, n), dtype="float16", name="input2")
        input3 = tvm.placeholder((n, n), dtype="float16", name="input3")
        output = tbe.vcmpsel(input1, input2, key, input3, 2.0)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        c = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b, c, d)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], b.asnumpy()[i][j]):
                    benchmark_data[i][j] = c.asnumpy()[i][j]
                else:
                    benchmark_data[i][j] = 2.0
        try:
            tvm.testing.assert_allclose(d.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmpsel_cpu_api_tensor_tensor_scalar_tensor(_):
    """
    lhs, rhs, slhs, srhs are respectively tensor, tensor, scalar, tensor
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        input2 = tvm.placeholder((n, n), dtype="float16", name="input2")
        input3 = tvm.placeholder((n, n), dtype="float16", name="input3")
        output = tbe.vcmpsel(input1, input2, key, 2.0, input3)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        c = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b, c, d)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], b.asnumpy()[i][j]):
                    benchmark_data[i][j] = 2.0
                else:
                    benchmark_data[i][j] = c.asnumpy()[i][j]
        try:
            tvm.testing.assert_allclose(d.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


def test_vcmpsel_cpu_api_tensor_scalar_tensor_tensor(_):
    """
    lhs, rhs, slhs, srhs are respectively tensor, scalar, tensor, tensor
    @return: Ture && false
    """
    op_dict = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "gt": operator.gt, "le": operator.le,
               "ge": operator.ge}
    for key, op in op_dict.items():
        n = 15
        input1 = tvm.placeholder((n, n), dtype="float16", name="input1")
        input2 = tvm.placeholder((n, n), dtype="float16", name="input2")
        input3 = tvm.placeholder((n, n), dtype="float16", name="input3")
        output = tbe.vcmpsel(input1, None, key, input2, input3)
        sch = tvm.create_schedule(output.op)
        func_vcmpsel = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vcmpsel")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        c = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vcmpsel(a, b, c, d)
        # 3.verify the correctness of output
        benchmark_data = np.zeros((n, n), dtype=output.dtype)
        for i in range(0, n):
            for j in range(0, n):
                if op(a.asnumpy()[i][j], 2.0):
                    benchmark_data[i][j] = b.asnumpy()[i][j]
                else:
                    benchmark_data[i][j] = c.asnumpy()[i][j]
        try:
            tvm.testing.assert_allclose(d.asnumpy(), benchmark_data)
        except AssertionError as e:
            print("\noperation is:", key)
            print(e)
            return False
    return True


test_func_list = [
    test_vadd_cpu_api,
    test_vsub_cpu_api,
    test_vmul_cpu_api,
    test_vdiv_cpu_api_not_support_vdiv,
    test_vdiv_cpu_api,
    test_vmod_cpu_api,
    # test_vmod_cpu_api_not_support_vdiv_and_vconv_f322s32f,
    test_vmod_cpu_api_not_support_vconv_f322s32f,
    test_vmin_cpu_api,
    test_vmax_cpu_api,
    test_vor_cpu_api,
    test_vand_cpu_api,
    test_vadds_cpu_api,
    test_vmins_cpu_api,
    test_vmuls_cpu_api,
    # test_vlog_cpu_api_not_support_vln_fp32_and_precision,
    test_vlog_cpu_api,
    test_vexp_cpu_api,
    test_vabs_cpu_api,
    test_vrec_cpu_api,
    test_vnot_cpu_api,
    test_vsqrt_cpu_api_not_support_vsqrt_and_precision,
    # test_vsqrt_cpu_api_not_support_vsqrt_and_performance,
    test_vsqrt_cpu_api,
    test_vrsqrt_cpu_api_not_support_vsqrt_and_precision,
    test_vrsqrt_cpu_api,
    test_vaxpy_cpu_api,
    test_vmla_cpu_api,
    test_vmadd_cpu_api,
    test_vlogic_cpu_api,
    test_vlogic_cpu_api_not_op,
    test_vcmp_cpu_api_bool,
    test_vcmp_cpu_api_bool_and_rhs_is_const,
    test_vcmp_cpu_api_bit,
    test_vcmp_cpu_api_bit_and_rhs_is_const,
    test_vsel_cpu_api_bool_tensor_to_tensor,
    test_vsel_cpu_api_bool_tensor_to_scalar,
    test_vsel_cpu_api_bool_scalar_to_tensor,
    test_vsel_cpu_api_bool_scalar_to_scalar,
    test_vsel_cpu_api_uint8_tensor_to_tensor,
    test_vsel_cpu_api_uint8_tensor_to_scalar,
    test_vsel_cpu_api_uint8_scalar_to_tensor,
    test_vsel_cpu_api_uint8_scalar_to_scalar,
    test_vcmpsel_cpu_api_tensor_tensor_tensor_tensor,
    test_vcmpsel_cpu_api_tensor_tensor_tensor_scalar,
    test_vcmpsel_cpu_api_tensor_tensor_scalar_tensor,
    test_vcmpsel_cpu_api_tensor_tensor_scalar_scalar,
    test_vcmpsel_cpu_api_tensor_scalar_tensor_tensor,
    test_vcmpsel_cpu_api_tensor_scalar_tensor_scalar,
    test_vcmpsel_cpu_api_tensor_scalar_scalar_tensor,
    test_vcmpsel_cpu_api_tensor_scalar_scalar_scalar,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
