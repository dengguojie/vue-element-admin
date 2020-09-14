import te.lang.cce
from te import tvm
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from topi import generic


SHAPE_SIZE_LIMIT = 2147483648


# Analysis parameter dim
def reduce_std_check_dim(shape_x, dim, keepdim):
    axis_dim = []
    ele_nums = 1
    if dim is None:
        if keepdim:
            raise TypeError
        for i in range(len(shape_x)):
            axis_dim.append(i)
            ele_nums = ele_nums * shape_x[i]
    elif isinstance(dim, int):
        axis_dim = dim
        ele_nums = shape_x[dim]
    else:
        for i in dim:
            axis_dim.append(i)
            ele_nums = ele_nums * shape_x[i]
    return axis_dim, ele_nums


@fusion_manager.register("reduce_std")
def reduce_std_compute(x, with_mean, dim, unbiased, keepdim,
                       kernel_name="reduce_std"):

    # Analysis parameter dim
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    dtype = x.dtype
    axis_dim, ele_nums = reduce_std_check_dim(shape_x, dim, keepdim)

    # calculate the number of elements
    one_const = tvm.const(1.0, dtype)
    nan_const = tvm.const(float("nan"), dtype)
    one_const_x_tensor = te.lang.cce.broadcast(one_const, shape_x)
    mean_sum = te.lang.cce.sum(one_const_x_tensor, axis=axis_dim, keepdims=True)

    # The number of elements cannot be greater than 1.
    mean_sum_return_tensor = te.lang.cce.sum(one_const_x_tensor, axis=axis_dim,
                                             keepdims=keepdim)

    nan_const_return_tensor = te.lang.cce.broadcast(nan_const, mean_sum_return_tensor.shape)
    x_return = te.lang.cce.reduce_max(x, axis=axis_dim, keepdims=keepdim)
    if ele_nums == 1:
        if with_mean:
            return [nan_const_return_tensor, x_return]
        else:
            return nan_const_return_tensor

    one_const_ms_tensor = te.lang.cce.broadcast(one_const, mean_sum.shape)
    mean_sum_minusone = te.lang.cce.vsub(mean_sum, one_const_ms_tensor)
    mean_cof_sca = te.lang.cce.vrec(mean_sum)
    mean_cof_sca_minusone = te.lang.cce.vrec(mean_sum_minusone)
    mean_cof = te.lang.cce.broadcast(mean_cof_sca, shape_x)
    mean_cof_minusone = te.lang.cce.broadcast(mean_cof_sca_minusone, shape_x)

    # calculate mu_muls
    mean_muls = te.lang.cce.vmul(x, mean_cof)

    # calculate mu
    mu = te.lang.cce.sum(mean_muls, axis=axis_dim, keepdims=True)

    # broadcast
    mu_broadcast = te.lang.cce.broadcast(mu, shape_x)

    # calculate x-mubroadcast
    x_mu_sub = te.lang.cce.vsub(x, mu_broadcast)

    # calculate x_mu_sub^2
    var_mul = te.lang.cce.vmul(x_mu_sub, x_mu_sub)

    # Divided by N or (N-1)
    if unbiased:
        var_muls = te.lang.cce.vmul(var_mul, mean_cof_minusone)
    else:
        var_muls = te.lang.cce.vmul(var_mul, mean_cof)

    # sum
    var = te.lang.cce.sum(var_muls, axis=axis_dim, keepdims=keepdim)

    # calculate the square root
    y = te.lang.cce.vsqrt(var)

    # Check whether the parameter with_mean is set to True.
    if with_mean:
        return [y, mu]
    else:
        return y


@util.check_input_type(dict, dict, dict, bool, (list, int), bool, bool, str)
def reduce_std(x, y1, y2, with_mean=False, dim=None, unbiased=True, keepdim=False,
               kernel_name="reduce_std"):

    # calculating data parameters
    check_list = ("float16", "float32")

    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    util.check_dtype_rule(dtype_x, check_list)
    util.check_shape_rule(shape_x)

    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = reduce_std_compute(data_x, with_mean, dim, unbiased, keepdim, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    if with_mean:
        config = {"name": kernel_name,
                  "tensor_list": [data_x] + list(res)}
    else:
        config = {"name": kernel_name,
                  "tensor_list": [data_x, res]}

    te.lang.cce.cce_build_code(schedule, config)
