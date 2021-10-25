# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from tbe.common.testing.testing import debug

warnings.filterwarnings("ignore")
ut_case = OpUT("nn_cpu", "dsl_cpu.test_nn_cpu_impl")


def test_vrelu_cpu_api(_):
    """
    The part less than 0 is taken as 0, and the part greater than 0 is tanken as the original value
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.vrelu(input1)
    sch = tvm.create_schedule(output.op)
    func_vrelu = tvm.build(sch, [input1, output], "c", "llvm", name="func_vrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vrelu(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.maximum(0, a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vmaddrelu_cpu_api(_):
    """
    relu(tensor0 * tensor2 + tensor1)
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    input3 = tvm.placeholder((n,), dtype="float16", name="input3")
    output = tbe.vmaddrelu(input1, input2, input3)
    sch = tvm.create_schedule(output.op)
    func_vmaddrelu = tvm.build(sch, [input1, input2, input3, output], "c", "llvm", name="func_vmaddrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=n).astype(input3.dtype), ctx)
    d = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vmaddrelu(a, b, c, d)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(d.asnumpy(), np.maximum(0, a.asnumpy() * c.asnumpy() + b.asnumpy()),
                                    atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vaddrelu_cpu_api_not_support_vaddrelu(_):
    """
    relu(tensor0 + tensor1)
    not supoort Intrinsic_vaddrelu
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    output = tbe.vaddrelu(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_vaddrelu = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vaddrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vaddrelu(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.maximum(0, a.asnumpy() + b.asnumpy()),
                                    atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vaddrelu_cpu_api(soc):
    """
    relu(tensor0 + tensor1)
    @param soc: soc version
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend710")
    output = tbe.vaddrelu(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_vaddrelu = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vaddrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vaddrelu(a, b, c)
    # 3.verify the correctness of output
    try:
        # Restore soc version to soc, or it affect the following use cases
        te_set_version(soc)
        tvm.testing.assert_allclose(c.asnumpy(), np.maximum(0, a.asnumpy() + b.asnumpy()),
                                    atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsubrelu_cpu_api_not_support_vsubrelu(_):
    """
    relu(tensor0 - tensor1)
    not support Intrinsic_vsubrelus
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    output = tbe.vsubrelu(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_vsubrelu = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vsubrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsubrelu(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.maximum(0, a.asnumpy() - b.asnumpy()),
                                    atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vsubrelu_cpu_api(soc):
    """
    relu(tensor0 - tensor1)
    @param soc: soc version
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    input2 = tvm.placeholder((n,), dtype="float16", name="input2")
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend710")
    output = tbe.vsubrelu(input1, input2)
    sch = tvm.create_schedule(output.op)
    func_vsubrelu = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func_vsubrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(input2.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vsubrelu(a, b, c)
    # 3.verify the correctness of output
    try:
        # Restore soc version to soc, or it affect the following use cases
        te_set_version(soc)
        tvm.testing.assert_allclose(c.asnumpy(), np.maximum(0, a.asnumpy() - b.asnumpy()),
                                    atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlrelu_cpu_api_not_support_vlrelu_and_int32(_):
    """
    not support Intrinsic_vlrelu and dtype is int32
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="int32", name="input1")
    output = tbe.vlrelu(input1)
    sch = tvm.create_schedule(output.op)
    func_vlrelu = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vlrelu(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.maximum(0.01 * a.asnumpy(), a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlrelu_cpu_api_not_support_vlrelu_and_alpha_is_const_0(_):
    """
    not support Intrinsic_vlrelu and alpha is zero
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    alpha = tvm.const(0, dtype="float16")
    output = tbe.vlrelu(input1, alpha)
    sch = tvm.create_schedule(output.op)
    func_vlrelu = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vlrelu(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.maximum(0.01 * a.asnumpy(), a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlrelu_cpu_api_not_support_vlrelu_and_alpha_is_const_1(_):
    """
    not support Intrinsic_vlrelu and alpha is one
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    alpha = tvm.const(1, dtype="float16")
    output = tbe.vlrelu(input1, alpha)
    sch = tvm.create_schedule(output.op)
    func_vlrelu = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vlrelu(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.maximum(0.01 * a.asnumpy(), a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlrelu_cpu_api_not_support_vlrelu_and_alpha_large_than_1(_):
    """
    not support Intrinsic_vlrelu and alpha large than 1
    @return: Ture && false
    """
    n = 10000
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    alpha = tvm.const(2, dtype="float16")
    output = tbe.vlrelu(input1, alpha)
    sch = tvm.create_schedule(output.op)
    func_vlrelu = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlrelu")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func_vlrelu(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.maximum(0.01 * a.asnumpy(), a.asnumpy()))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_vlrelu_cpu_api(soc):
    """
    use tvm.lrelu
    @param soc: soc version
    @return: Ture && false
    """
    with debug():
        n = 10000
        input1 = tvm.placeholder((n,), dtype="float16", name="input1")
        from te.platform.cce_conf import te_set_version
        te_set_version("Ascend710")
        output = tbe.vlrelu(input1)
        sch = tvm.create_schedule(output.op)
        func_vlrelu = tvm.build(sch, [input1, output], "c", "llvm", name="func_vlrelu")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func_vlrelu(a, b)
        # 3.verify the correctness of output
        try:
            # Restore soc version to soc, or it affect the following use cases
            te_set_version(soc)
            tvm.testing.assert_allclose(b.asnumpy(), np.maximum(0.01 * a.asnumpy(), a.asnumpy()))
        except AssertionError as e:
            print(e)
            return False
        return True


def test_round_to_cpu_api(_):
    """
    use tvm.lrelu
    @return: Ture && false
    """
    n = 1024
    max_value, min_value = 0.8, 0.5
    input1 = tvm.placeholder((n,), dtype="float16", name="input1")
    output = tbe.round_to(input1, max_value, min_value)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=n).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    min_value_array = np.full((n, ), min_value, dtype=input1.dtype)
    max_value_array = np.full((n, ), max_value, dtype=input1.dtype)
    benchmark_data = np.maximum(a.asnumpy(), min_value_array)
    benchmark_data = np.minimum(benchmark_data, max_value_array)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), benchmark_data)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_broadcast_cpu_api_var_is_tensor(_):
    """
    for broadcast api
    @return: Ture && false
    """
    out_shape = (1024, 1024)
    shape = (1, 1024)
    input1 = tvm.placeholder(shape, dtype="float16", name="input1")
    output = tbe.broadcast(input1, out_shape)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=shape).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(out_shape, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(b.asnumpy(), np.broadcast_to(a.asnumpy(), out_shape))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_broadcast_cpu_api_var_is_not_tensor(_):
    """
    for broadcast api
    @return: Ture && false
    """
    out_shape = (1024, 1024)
    shape = 1
    input1 = tvm.const(shape, dtype="float16")
    output = tbe.broadcast(input1, out_shape)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.zeros(out_shape, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(a.asnumpy(), np.broadcast_to(shape, out_shape))
    except AssertionError as e:
        print(e)
        return False
    return True


test_func_list = [
    test_vrelu_cpu_api,
    test_vmaddrelu_cpu_api,
    test_vaddrelu_cpu_api_not_support_vaddrelu,
    test_vaddrelu_cpu_api,
    test_vsubrelu_cpu_api_not_support_vsubrelu,
    test_vsubrelu_cpu_api,
    test_vlrelu_cpu_api_not_support_vlrelu_and_int32,
    test_vlrelu_cpu_api_not_support_vlrelu_and_alpha_is_const_0,
    test_vlrelu_cpu_api_not_support_vlrelu_and_alpha_is_const_1,
    test_vlrelu_cpu_api_not_support_vlrelu_and_alpha_large_than_1,
    test_vlrelu_cpu_api,
    test_round_to_cpu_api,
    test_broadcast_cpu_api_var_is_tensor,
    test_broadcast_cpu_api_var_is_not_tensor,
]
support_soc = ["Ascend310", "Ascend910A"]
for item in test_func_list:
    ut_case.add_cust_test_func(support_soc=support_soc, test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
