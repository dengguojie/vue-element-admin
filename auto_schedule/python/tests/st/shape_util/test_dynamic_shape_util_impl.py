from sch_test_frame.ut import OpUT
from tbe.common.utils import shape_util
from tbe.tvm import expr as _expr


ut_case = OpUT("shape_util", "shape_util.test_dynamic_shape_util_impl")

def test_squeeze_shape(_):
    result = shape_util.squeeze_shape([1, 1, 1])
    return len(result) == 1 and result[0] == 1

def test_wrap_axes_to_positive_rank_error(_):
    try:
        shape_util.wrap_axes_to_positive((1, 2, 100,), 10)
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_wrap_axes_to_positive_list(_):
    result = shape_util.wrap_axes_to_positive((1, 2, -7,), 10)
    return [1, 2, 3] == result

def test_wrap_axes_to_positive_int(_):
    result = shape_util.wrap_axes_to_positive(1, 10)
    return [1,] == result

def test_refine_shape_axes_1(_):
    shape, axis = shape_util.refine_shape_axes((2,3,4,5,1,6), (1, -4))
    return shape == [2, 12, 30] and axis == [1,]

def test_refine_shape_axes_2(_):
    shape, axis = shape_util.refine_shape_axes((2,), (0,))
    return shape == (2,) and axis == (0,)

def test_refine_shape_axes_3(_):
    shape, axis = shape_util.refine_shape_axes((1, 1, 1), (0,))
    return shape == [1,] and axis == []

def test_unify_broadcast_shapes_error(_):

    try:
        shape_util.unify_broadcast_shapes( [[2, 3], [3, 2, 5], [3, 1, 3]], "ADD_N")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_unify_broadcast_shapes1(_):
    shapes = shape_util.unify_broadcast_shapes( [[2, 3], [3, 2, 1], [3, 1, 3]], "ADD_N")
    return shapes == ([1, 2, 3], [3, 2, 1], [3, 1, 3], [3, 2, 3] )

def test_unify_broadcast_shapes2(_):
    a1 = _expr.IntImm("int8", 1)
    a2 = _expr.IntImm("int8", 2)
    a3 = _expr.IntImm("int8", 1)
    shapes = shape_util.unify_broadcast_shapes( [[a1, a2, a3], [3, 1, 4],], "ADD_N")
    print(shapes)
    return shapes == ([a1, a2, a3], [3, 1, 4], [3, a2, 4])

def test_unify_broadcast_shapes3(_):
    shapes = shape_util.unify_broadcast_shapes( [[1, 1, 1], [3, 2, 1],], "ADD_N")
    return shapes == ([1, 1, 1], [3, 2, 1], [3, 2, 1])

def test_broadcast_shapes(_):
    shape1, shape2, out_shape = shape_util.broadcast_shapes([2, 3], [3, 2, 1],
                                                            "broadcast_shapes", "input_x", "input_y")
    return shape1 == [1, 2, 3] and shape2 == [3, 2, 1] and out_shape == [3, 2, 3]

def test_broadcast_shapes_error(_):
    try:
        shape_util.broadcast_shapes([2, 3], [3, 2, 5],
                                    "broadcast_shapes", "input_x", "input_y")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_refine_shapes_for_broadcast(_):
    shape1, shape2 = shape_util.refine_shapes_for_broadcast([2, 3], [3, 2, 1])
    return shape1 == [1, 2, 3] and shape2 == [3, 2, 1]

def test_refine_shapes_for_broadcast_empty(_):
    shape1, shape2 = shape_util.refine_shapes_for_broadcast([], [])
    return shape1 == [1,] and shape2 == [1,]

def test_refine_shapes_for_broadcast_fuse(_):
    shape1, shape2 = shape_util.refine_shapes_for_broadcast([4, 5, 3, 1, 1, 3], [4, 5, 1, 3, 3, 1])
    return shape1 == [20, 3, 1, 3] and shape2 == [20, 1, 9, 1]

def test_simplify_axis_shape1(_):
    shape, axis = shape_util.simplify_axis_shape([2, 3, 4, 5, 6], [0 , 1, 4])
    return shape == [6, 20, 6] and axis == [0 ,2]

def test_simplify_axis_shape2(_):
    shape, axis = shape_util.simplify_axis_shape([2, 3, 4, 5, 6], [0 , 1,])
    return shape == [6, 120] and axis == [0,]

def test_shape_refine_no_reduce(_):
    res_shape = shape_util.shape_refine([2, 3, 4, 5, 6, 1])
    return res_shape == [2, 3, 4, 5, 6,]

def test_shape_refine_no_reduce_all_one(_):
    res_shape = shape_util.shape_refine([1, 1, 1,])
    return res_shape == [1,]

def test_shape_refine_reduce_list(_):
    res_shape, res_axis = shape_util.shape_refine([2, 3, 1, 5, 6, 10], [0, 1, 4, 5])
    return res_shape == [2, 3, 5, 6, 10] and res_axis ==  [0, 1, 3, 4]

def test_shape_refine_reduce_int(_):
    res_shape, res_axis = shape_util.shape_refine([2, 3, 1, 5, 6, 10], 0)
    return res_shape == [2, 3, 5, 6, 10] and res_axis ==  [0,]

def test_shape_refine_reduce_reduce_1(_):
    res_shape, res_axis = shape_util.shape_refine([2, 3, 1, 5, 6, 10], reduce_axis=2)
    return res_shape == [2, 3, 1, 5, 6, 10] and res_axis ==  [2,]

def test_shape_refine_reduce_reduce_2(_):
    res_shape, res_axis = shape_util.shape_refine([2, 3, 1, 5, 6, 10], reduce_axis=[2,])
    return res_shape == [2, 3, 1, 5, 6, 10] and res_axis ==  [2,]

def test_shape_refine_reduce_reduce_3(_):
    res_shape, res_axis = shape_util.shape_refine([2, 3, 1, 5, 6, 10], reduce_axis=[2,], keep_dims=False)
    return res_shape == [2, 3, 1, 5, 6, 10] and res_axis ==  [2,]

def test_shape_refine_reduce_reduce_4(_):
    res_shape, res_axis = shape_util.shape_refine([2, 3, 1, 5, 6, 10], reduce_axis=[1,], keep_dims=False)
    return res_shape == [2, 3, 5, 6, 10] and res_axis ==  [1,]

def test_refine_axis_error(_):
    try:
        shape_util.refine_axis([-2, 11], [2, 3, 1, 5, 6, 10])
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_axis_check_int(_):
    res_axis = shape_util.axis_check(10, -1)
    print(res_axis)
    return res_axis == 9

def test_axis_check_int_error(_):
    try:
        shape_util.axis_check(10, "-1")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_axis_check_list(_):
    res_axis = shape_util.axis_check(10, [-1,])
    print(res_axis)
    return res_axis == [9,]

def test_axis_check_list_error(_):
    try:
        shape_util.axis_check(10, [-100,])
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_produce_shapes(_):
    shape1, shape2, out_shape = shape_util.produce_shapes([2, 3], [3, 2, 1])
    return shape1 == [1, 2, 3] and shape2 == [3, 2, 1] and out_shape == [3, 2, 3]

def test_scalar2tensor_one(_):
    shape1 = shape_util.scalar2tensor_one([2, 3])
    shape2 = shape_util.scalar2tensor_one([])
    print(shape1)
    print(shape2)
    return shape1 == [2, 3] and shape2 == [1]

def test_axis_transform_5d_NCHW_neg(_):
    return shape_util.axis_transform_5d( -1, "NCHW") == -2

def test_axis_transform_5d_NCHW(_):
    return shape_util.axis_transform_5d( 1, "NCHW") == 1

def test_axis_transform_5d_NHWC_neg4(_):
    return shape_util.axis_transform_5d( -4, "NHWC") == -5

def test_axis_transform_5d_NHWC_neg1(_):
    return shape_util.axis_transform_5d( -1, "NHWC") == -4

def test_axis_transform_5d_NHWC_1(_):
    return shape_util.axis_transform_5d( 1, "NHWC") == 2

def test_axis_transform_5d_NHWC_2(_):
    return shape_util.axis_transform_5d( 2, "NHWC") == 3

def test_axis_transform_5d_NHWC_3(_):
    return shape_util.axis_transform_5d( 3, "NHWC") == 1

def test_compare_tensor_dict_key_error1(_):
    try:
        shape_util.compare_tensor_dict_key("1", {"dtype":"float16"}, "dtype")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_compare_tensor_dict_key_error2(_):
    try:
        shape_util.compare_tensor_dict_key({"dtype":"float16"}, "1", "dtype")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_compare_tensor_dict_key_error3(_):
    try:
        shape_util.compare_tensor_dict_key({"dtype":"float16"}, {"dtype":"float16"}, "shape")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_compare_tensor_dict_key_error4(_):
    try:
        shape_util.compare_tensor_dict_key({"dtype":"float16", "shape":[1, 2]}, {"dtype":"float16"}, "shape")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_compare_tensor_dict_key_error5(_):
    try:
        shape_util.compare_tensor_dict_key({"dtype":"float16", "shape":[1, 2]}, {"dtype":"float16", "shape":"[1, 2]"}, "shape")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_compare_tensor_dict_key_error6(_):
    try:
        shape_util.compare_tensor_dict_key({"dtype":"float16", "shape":[1, 2]}, {"dtype":"float32", "shape":[1, 2]}, "dtype")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_compare_tensor_dict_key_error7(_):
    try:
        shape_util.compare_tensor_dict_key({"dtype":"float16", "shape":[1, 2]}, {"dtype":"float32", "shape":[1, 3]}, "shape")
    except RuntimeError as e:
        print(e)
        return True
    return False

def test_get_shape_size(_):
    return shape_util.get_shape_size([1, 2, 3]) == 6


case_list = [
    test_squeeze_shape,
    test_wrap_axes_to_positive_rank_error,
    test_wrap_axes_to_positive_list,
    test_wrap_axes_to_positive_int,
    test_refine_shape_axes_1,
    test_refine_shape_axes_2,
    test_refine_shape_axes_3,
    test_unify_broadcast_shapes_error,
    test_unify_broadcast_shapes1,
    test_unify_broadcast_shapes2,
    test_unify_broadcast_shapes3,
    test_broadcast_shapes,
    test_broadcast_shapes_error,
    test_refine_shapes_for_broadcast,
    test_refine_shapes_for_broadcast_empty,
    test_refine_shapes_for_broadcast_fuse,
    test_simplify_axis_shape1,
    test_simplify_axis_shape2,
    test_shape_refine_no_reduce,
    test_shape_refine_no_reduce_all_one,
    test_shape_refine_reduce_list,
    test_shape_refine_reduce_int,
    test_shape_refine_reduce_reduce_1,
    test_shape_refine_reduce_reduce_2,
    test_shape_refine_reduce_reduce_3,
    test_shape_refine_reduce_reduce_4,
    test_refine_axis_error,
    test_axis_check_int,
    test_axis_check_int_error,
    test_axis_check_list,
    test_axis_check_list_error,
    test_produce_shapes,
    test_scalar2tensor_one,
    test_axis_transform_5d_NCHW_neg,
    test_axis_transform_5d_NCHW,
    test_axis_transform_5d_NHWC_neg4,
    test_axis_transform_5d_NHWC_neg1,
    test_axis_transform_5d_NHWC_1,
    test_axis_transform_5d_NHWC_2,
    test_axis_transform_5d_NHWC_3,
    test_compare_tensor_dict_key_error1,
    test_compare_tensor_dict_key_error2,
    test_compare_tensor_dict_key_error3,
    test_compare_tensor_dict_key_error4,
    test_compare_tensor_dict_key_error5,
    test_compare_tensor_dict_key_error6,
    test_compare_tensor_dict_key_error7,
    test_get_shape_size,
]
for item in case_list:
    ut_case.add_cust_test_func(test_func=item)
