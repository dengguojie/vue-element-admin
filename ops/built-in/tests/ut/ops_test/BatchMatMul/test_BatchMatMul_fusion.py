import functools

import te.platform as tbe_platform
from batchmatmul_fusion_case import batchmatmul_ut_fusion_case
from tbe.dsl.static_schedule import cce_build_code
from te.tvm.target import cce
from topi.generic import auto_schedule
from te import tvm
from te.utils import shape_util
from impl.batch_matmul import batch_matmul_compute
from impl.batch_matmul import _get_input_shape
from impl.batch_matmul import _get_bias
from impl.reduce_sum_d import reduce_sum_d_compute
from impl.fused_mul_add import fusion_mul_add_compute
from impl.fused_mul_add import check_format
from impl.fused_mul_add import _infer_shape_one
from impl.fused_mul_add import _infer_shape_two
from impl.add_n import add_n_compute_for_fusion
from impl.fast_gelu_grad import fast_gelu_grad_compute
from te.utils import para_check

def _get_batchmatmul_node(case):
    """
    get out put node of batchmatmul
    """
    input_x,input_y,bias,output_z,trans_a,trans_b = case[:6]

    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    shape_a_length = len(shape_a)
    shape_b_length = len(shape_b)
    if shape_a is not None:
        if shape_a_length < 2:
            shape_a = input_x.get("shape")

    if shape_b is not None:
        if shape_b_length < 2:
            shape_b = input_y.get("shape")
    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        if input_x.get("format") == "FRACTAL_NZ":
            shape_bias = _get_bias(shape_bias)

    src_dtype = input_x.get("dtype").lower()
    dst_dtype = output_z.get("dtype").lower()
    is_fractal = False

    shape_a = list(shape_a)
    shape_b = list(shape_b)
    if input_x.get("format") == "FRACTAL_NZ":
        shape_a = _get_input_shape(shape_a)
        shape_b = _get_input_shape(shape_b)

    trans_a_local = trans_a
    trans_b_local = trans_b

    if input_x.get("format") == "FRACTAL_NZ":
        batch_axis = shape_a[:(len(shape_a) - 2)]
        shape_a = batch_axis + [shape_a[len(shape_a) - 1], shape_a[len(shape_a) - 2]]
        trans_a_local = bool(1 - trans_a)

    if input_y.get("format") == "FRACTAL_NZ":
        batch_axis = shape_b[:(len(shape_b) - 2)]
        shape_b = batch_axis + [shape_b[len(shape_b) - 1], shape_b[len(shape_b) - 2]]
        trans_b_local = bool(1 - trans_b)

    inp_src_dtype = src_dtype.lower()
    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_b) - 2]
    n_shape = shape_b[len(shape_b) - 1]

    if inp_src_dtype == "float16":
        block_reduce = tbe_platform.BLOCK_REDUCE

    block_in = tbe_platform.BLOCK_IN
    block_out = tbe_platform.BLOCK_OUT

    if trans_a:
        shape_a_dup = (m_shape // block_reduce, km_shape // block_in, block_reduce, block_in)
    else:
        shape_a_dup = (m_shape // block_in, km_shape // block_reduce, block_in, block_reduce)

    if trans_b:
        shape_b_dup = (kn_shape // block_out, n_shape // block_reduce, block_reduce, block_out)
    else:
        shape_b_dup = (kn_shape // block_reduce, n_shape // block_out, block_out, block_reduce)

    if input_x.get("format") == "FORMAT_FRACTAL_Z":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "fractal"
    elif input_x.get("format") == "FRACTAL_NZ":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "FRACTAL_NZ"
    else:
        shape_a_dup = (shape_a[len(shape_a) - 2], shape_a[len(shape_a) - 1])
        format_a = "ND"

    if input_y.get("format") == "FORMAT_FRACTAL_Z":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "fractal"
    elif input_y.get("format") == "FRACTAL_NZ":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "FRACTAL_NZ"
    else:
        shape_b_dup = (shape_b[len(shape_b) - 2], shape_b[len(shape_b) - 1])
        format_b = "ND"

    batch_shape_a = None
    if len(shape_a) > 2:
        batch_shape_a = functools.reduce(lambda x, y: x * y, shape_a[:-2])

    batch_shape_b = None
    if len(shape_b) > 2:
        batch_shape_b = functools.reduce(lambda x, y: x * y, shape_b[:-2])

    if len(shape_a) >= len(shape_b):
        batch_shape = batch_shape_a
    else:
        batch_shape = batch_shape_b

    if batch_shape is not None and batch_shape >= 1:
        if is_fractal:
            if batch_shape_a is not None:
                shape_a_dup = (batch_shape_a,) + shape_a_dup
            if batch_shape_b is not None:
                shape_b_dup = (batch_shape_b,) + shape_b_dup
        else:
            if batch_shape_a is not None:
                shape_a_dup = (batch_shape_a,) + shape_a_dup
            if batch_shape_b is not None:
                shape_b_dup = (batch_shape_b,) + shape_b_dup

    tensor_bias = None
    shape_bias_length = len(shape_bias)
    if shape_bias_length <= 2:
        shape_bias_dup = shape_bias
    else:
        shape_bias_dup = (shape_bias[len(shape_bias) - 2], shape_bias[len(shape_bias) - 1])
        bias_batch_size = functools.reduce(lambda x, y: x * y, shape_bias[:-2])
        shape_bias_dup = (bias_batch_size,) + shape_bias_dup

    tensor_a = tvm.placeholder(shape_a_dup, name='tensor_a',
                               attrs={'format': format_a,
                               "ori_shape": input_x.get("ori_shape") },
                               dtype=inp_src_dtype)
    tensor_b = tvm.placeholder(shape_b_dup, name='tensor_b',
                               attrs={'format': format_b,
                               "ori_shape": input_y.get("ori_shape")},
                               dtype=inp_src_dtype)

    if shape_bias_length > 0:
        tensor_bias = tvm.placeholder(shape_bias_dup, name='tensor_bias',
                                      dtype=dst_dtype)
    result = batch_matmul_compute(tensor_a, tensor_b, tensor_bias,
                                       output_z, trans_a, trans_b)
    tensor_list = [tensor_a, tensor_b, result]

    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    return result, tensor_list

def test_batchmatmul_reduce_sum_fusion_ub(fusion_case):
        with cce():
            outs, tensor_list = _get_batchmatmul_node(fusion_case)
            y = fusion_case[6]
            axis = fusion_case[7]
            keepdims = None
            x = fusion_case[3]
            shape = x.get("shape")
            dtype = x.get("dtype")
            format_x = x.get("format")
            format_y = y.get("format")
            format_ori_y = y.get("ori_format")
            dtype_lower = dtype.lower()

            axis_d = []
            shape_len = len(shape)
            if not axis:
                for i, _ in enumerate(shape):
                    axis_d.append(i)
            else:
                axis_d = list(axis)
            axis_d = shape_util.axis_check(shape_len, axis_d)
            # 5HD Special param for 5hd schedule
            is_nz_nd = False
            if format_x == "FRACTAL_NZ" and format_ori_y == format_y:
                is_nz_nd = True
            is_5hdc = para_check.check_and_init_5hdc_reduce_support(x, axis)

            if not keepdims and not is_5hdc:
                shape, axis_d = shape_util.shape_refine(list(shape), axis_d, keepdims)
                shape, axis_d = shape_util.simplify_axis_shape(shape, axis_d)

            res = reduce_sum_d_compute(outs, y, axis_d, keepdims,
                                        is_5hdc=is_5hdc, is_nz_nd=is_nz_nd)
            if is_5hdc:
                res.ori_shape = x["ori_shape"]
                res.ori_format = x["ori_format"]

            tensor_list.append(res)
            sch = auto_schedule(res)
            config = {
                "print_ir":False,
                "need_build":True,
                "name":"batchmatmul_reducesum_fusion",
                "tensor_list":tensor_list,
            }
            cce_build_code(sch, config)


def test_batchmatmul_fusedmuladd_fusion_ub(fusion_case):
    with cce():
        result_matmul, tensor_list = _get_batchmatmul_node(fusion_case)
        output = fusion_case[3]
        mul = fusion_case[6]
        add = fusion_case[7]
        shape_input0 = list(shape_util.scalar2tensor_one(output.get("shape")))
        shape_input1 = list(shape_util.scalar2tensor_one(mul.get("shape")))
        shape_input2 = list(shape_util.scalar2tensor_one(add.get("shape")))

        dtype_input0 = output.get("dtype").lower()
        dtype_input1 = mul.get("dtype").lower()
        dtype_input2 = add.get("dtype").lower()

        format_input0 = output.get("format").upper()
        format_input1 = mul.get("format").upper()
        format_input2 = add.get("format").upper()

        format_pattern = check_format(format_input0, format_input1, format_input2)
        if format_pattern in [1, 2, 3]:
            shape_input0, shape_input1, shape_input2 = \
                _infer_shape_one(shape_input0, shape_input1,
                                shape_input2, format_pattern)
        elif format_pattern == 4:
            shape_input0, shape_input1, shape_input2 = \
                _infer_shape_two(shape_input0, shape_input1,
                                shape_input2, format_pattern)
        else:
            shape_input0, shape_input1, shape_max_mul = \
                shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                            param_name_input2="input1")
            shape_input2, shape_max_mul, shape_max_add0 = \
                shape_util.broadcast_shapes(shape_input2, shape_max_mul, param_name_input1="input2",
                                            param_name_input2="shape_max_mul")

        data_input0 = tvm.placeholder(shape_input0,
                                    name="data_input0",
                                    attrs={'format': format_input0},
                                    dtype=dtype_input0)
        data_input1 = tvm.placeholder(shape_input1,
                                    name="data_input1",
                                    attrs={'format': format_input1},
                                    dtype=dtype_input1)
        data_input2 = tvm.placeholder(shape_input2,
                                    name="data_input2",
                                    attrs={'format': format_input2},
                                    dtype=dtype_input2)
        result_matmul.op.tag = "matmul"
        res = fusion_mul_add_compute(result_matmul, data_input1, data_input2, [])

        tensor_list += [data_input0, data_input1, data_input2, res]
        sch = auto_schedule(res)
        config = {
            "print_ir":False,
            "need_build":True,
            "name":"batchmatmul_reducesum_fusion",
            "tensor_list":tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_addn_fusion_ub(fusion_case):
    with cce():
        result_matmul, tensor_list = _get_batchmatmul_node(fusion_case)
        output = fusion_case[3]
        addn = fusion_case[6]
        shape_input0 = list(shape_util.scalar2tensor_one(output.get("shape")))
        shape_input1 = list(shape_util.scalar2tensor_one(addn.get("shape")))

        dtype_input0 = output.get("dtype").lower()
        dtype_input1 = addn.get("dtype").lower()

        format_input0 = output.get("format").upper()
        format_input1 = addn.get("format").upper()

        data_input0 = tvm.placeholder(shape_input0,
                                      name="data_input0",
                                      attrs={'format': format_input0},
                                      dtype=dtype_input0)
        data_input1 = tvm.placeholder(shape_input1,
                                      name="data_input1",
                                      attrs={'format': format_input1},
                                      dtype=dtype_input1)

        result_matmul.op.tag = "matmul"
        res = add_n_compute_for_fusion([result_matmul, data_input1], data_input0, 2)

        tensor_list += [data_input1, res]
        sch = auto_schedule(res)
        config = {
            "print_ir":False,
            "need_build":True,
            "name":"batchmatmul_addn_fusion",
            "tensor_list":tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_fast_gelu_grad_fusion_ub(fusion_case):
    with cce():
        result_matmul, tensor_list = _get_batchmatmul_node(fusion_case)
        output = fusion_case[3]
        input_x = fusion_case[6]
        shape_input0 = list(shape_util.scalar2tensor_one(output.get("shape")))
        shape_input1 = list(shape_util.scalar2tensor_one(input_x.get("shape")))

        dtype_input0 = output.get("dtype").lower()
        dtype_input1 = input_x.get("dtype").lower()

        format_input0 = output.get("format").upper()
        format_input1 = input_x.get("format").upper()

        data_input0 = tvm.placeholder(shape_input0,
                                      name="data_input0",
                                      attrs={'format': format_input0},
                                      dtype=dtype_input0)
        data_input1 = tvm.placeholder(shape_input1,
                                      name="data_input1",
                                      attrs={'format': format_input1},
                                      dtype=dtype_input1)

        result_matmul.op.tag = "matmul"
        res = fast_gelu_grad_compute(result_matmul, data_input1, data_input0)
        res_other_format = fast_gelu_grad_compute(data_input1, result_matmul, data_input0)

        tensor_list += [data_input1, res, res_other_format]
        sch = auto_schedule(res, res_other_format)
        config = {
            "print_ir":False,
            "need_build":True,
            "name":"batchmatmul_fast_gelu_grad_fusion",
            "tensor_list":tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_fusion(fusion_case):
    def test_fusion_case(test_args):
        print(fusion_case.get("case_name"))
        if "reduce_sum" in fusion_case.get("case_name"):
            test_batchmatmul_reduce_sum_fusion_ub(fusion_case.get("params"))
        elif "fused_mul_add" in fusion_case.get("case_name"):
            test_batchmatmul_fusedmuladd_fusion_ub(fusion_case.get("params"))
        elif "addn" in fusion_case.get("case_name"):
            test_batchmatmul_addn_fusion_ub(fusion_case.get("params"))
        elif "fast_gelu_grad" in fusion_case.get("case_name"):
            test_batchmatmul_fast_gelu_grad_fusion_ub(fusion_case.get("params"))

    return test_fusion_case