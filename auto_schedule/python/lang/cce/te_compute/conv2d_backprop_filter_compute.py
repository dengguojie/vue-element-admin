# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd.

conv2d backprop filter DSL interface.

"""
from __future__ import absolute_import
from __future__ import print_function
import functools
from te import tvm
from te.lang.cce.te_compute import util
from te.platform import intrinsic_check_support
from te.platform.cce_conf import CceProductParams as pver
from te.utils.error_manager import error_manager_util as err_man

# fractal size, only support 16 for now
BLOCK_SIZE = 16
# maximum of int64 (2**63 - 1)
DATA_SIZE_LIMIT_INT64 = 9223372036854775807
# maximum of w in conv1d is (2**31 - 1)
CONV1D_MAX_W = 2147483647

def check_shape_rule(shape, dim, formats, name, allow_zero=False):
    """
    check shape

    """
    if len(shape) != dim:
        dict_args = dict()
        dict_args["errCode"] = "E64003"
        dict_args["param_name"] = name
        dict_args["format"] = formats
        dict_args["expect_dim"] = str(dim)
        dict_args["dim"] = str(len(shape))
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))
    axis_num = 0
    for dim_x in shape:
        check_zero = False
        if allow_zero:
            check_zero = (dim_x < 0)
            axis_rule = "int and >= 0 "
        else:
            check_zero = (dim_x <= 0)
            axis_rule = "int and > 0"
        if (not isinstance(dim_x, int)) or check_zero:
            dict_args = dict()
            dict_args["errCode"] = "E64004"
            dict_args["param_name"] = name
            dict_args["axis_rule"] = axis_rule
            dict_args["wrong_axis"] = str(axis_num)
            dict_args["actual_value"] = str(dim_x)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        axis_num = axis_num + 1


def ceil_div(dividend, divisor):
    """
    do division and round up to an integer

    """
    if divisor == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))
    return (dividend + divisor - 1) // divisor


def check_attr_rule(attr, dim, attr_limit, formats, name):
    """
    check attribute

    """
    attr_min = attr_limit[0]
    attr_max = attr_limit[1]
    if len(attr) != dim:
        dict_args = dict()
        dict_args["errCode"] = "E64003"
        dict_args["param_name"] = name
        dict_args["format"] = formats
        dict_args["expect_dim"] = str(dim)
        dict_args["dim"] = str(len(attr))
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))
    for attr_x in attr:
        if (not isinstance(attr_x, int)) \
                or attr_x < attr_min or attr_x > attr_max:
            dict_args = {}
            dict_args['errCode'] = "E64001"
            dict_args['range'] = "[{},{}]".format(attr_min, attr_max)
            dict_args['attr_name'] = name
            dict_args["value"] = str(attr_x)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))


def check_variable_range(variable, minimum, maximum, name):
    """
    check variable range

    """
    if (not isinstance(variable, int)) or variable < minimum \
            or variable > maximum:
        dict_args = dict()
        dict_args['errCode'] = "E64001"
        dict_args['range'] = "[{},{}]".format(minimum, maximum)
        dict_args['attr_name'] = name
        dict_args["value"] = str(variable)
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))


def check_addressing_rule(shape, byte_count, limit, name):
    """
    check addressing limit

    """
    # product of all dimension
    product = functools.reduce(lambda x, y: x * y, shape[:])
    if product*byte_count > limit:
        dict_args = dict()
        dict_args['errCode'] = "E60020"
        dict_args['attr_name'] = name
        raise RuntimeError(dict_args,
                           err_man.get_error_message(dict_args))


class Conv2dBackpropFilter:  # pylint: disable=R0902
    """
    Conv2dBackpropFilter: compute definition of conv2d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    deconv_dw_input_check_1: parameters check part 1

    deconv_dw_input_check_2: parameters check part 2

    deconv_dw_compute : compute process

    grads_2_matrix : compute definition of loading grads to cbuf

    grads_2_fractal : compute definition of load_2d

    fmap_2_col_matrix : compute definition of set_fmatrix

    fmap_2_col_fractal : compute definition of load_3d

    mad : compute definition of mmad

    """
    def __init__(self, input_x, out_backprop,  # pylint: disable=R0913, R0914
                 filter_sizes, strides, padding, dilations,
                 res_dtype="float32",
                 kernel_name="conv2d_backprop_filter_cce"):
        """
        initialization

        Parameters:
        ----------
        input_x : the featuremap data, tvm.placeholder, 5HD shape

        out_backprop : the grads data, tvm.placeholder, 5HD shape

        filter_sizes : 4-D shape, specifies the filter sizes

        strides : 2-D shape in height and width dimension

        padding : 4-D shape in up/down/left/right dimension

        dilations : 4-D shape in batch/channel/height/width dimension

        res_dtype : the output data type

        Returns
        -------
        None
        """

        self.shapelist = {}

        self.fmap = input_x
        self.grads = out_backprop

        self.weight_shape = list(filter_sizes)

        self.fmap_dtype = input_x.dtype
        self.grads_dtype = out_backprop.dtype
        self.res_dtype = res_dtype

        self.pad = list(padding)
        self.stride = list(strides)
        self.dilation = list(dilations)
        self.kernel_name = kernel_name

        self.optag = "conv2d_backprop_filter"

        # 5HD shape
        self.shape_grads_5hd = list(x.value for x in self.grads.shape)
        self.shape_x_5hd = list(x.value for x in self.fmap.shape)

        self.shapelist['grads_5hd'] = self.shape_grads_5hd
        self.shapelist['fmap_5hd'] = self.shape_x_5hd

        self.dw_ddr = []
        self.res_tensor = self.dw_ddr  # return tensor of this file to topi

        # flag of special case
        self.flag_all_one_case = False
        self.flag_load3d_special_case = False
        self.conv1d_situation = False


    def deconv_dw_input_check_1(self):

        """
        do input parameters check part1

        """
        # check of data type
        if self.fmap_dtype != "float16":
            dict_args = dict()
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "fmap_dtype"
            dict_args["expected_dtype_list"] = "float16"
            dict_args["dtype"] = self.fmap_dtype
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        if self.grads_dtype != "float16":
            dict_args = dict()
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "grads_dtype"
            dict_args["expected_dtype_list"] = "float16"
            dict_args["dtype"] = self.grads_dtype
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        if not intrinsic_check_support("Intrinsic_mmad", "f162f32") and \
                self.res_dtype != "float16":
            dict_args = dict()
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "res_dtype"
            dict_args["expected_dtype_list"] = "float16 for lhisi"
            dict_args["dtype"] = self.res_dtype
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        if intrinsic_check_support("Intrinsic_mmad", "f162f32") and \
                self.res_dtype != "float32":
            dict_args = dict()
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "res_dtype"
            dict_args["expected_dtype_list"] = "float32"
            dict_args["dtype"] = self.res_dtype
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        # check shape
        # each element must be positive int
        check_shape_rule(self.shape_grads_5hd, 5, "NC1HWC0", "out_backprop")
        check_shape_rule(self.shape_x_5hd, 5, "NC1HWC0", "x")
        check_shape_rule(self.weight_shape, 4, "NCHW", "filter_sizes")
        check_shape_rule(self.pad, 4, "up_bottom_left_right", "pad",
                         allow_zero=True)
        check_shape_rule(self.stride, 2, "height_weight", "stride")
        check_shape_rule(self.dilation, 4, "NCHW", "dilation")

        _, _, grads_height, grads_width, _ \
            = self.shape_grads_5hd
        _, _, fmap_height, fmap_width, _ = self.shape_x_5hd
        _, _, kernel_height, kernel_width = self.weight_shape
        pad_top, pad_bottom, pad_left, pad_right = self.pad
        stride_height, _ = self.stride
        _, _, dilationh, _ = self.dilation

        fmap_height_after_pad = fmap_height + pad_top + pad_bottom
        fmap_width_after_pad = fmap_width + pad_left + pad_right

        if ceil_div(self.weight_shape[0], 16) != self.shape_grads_5hd[1]:
            dict_args = {}
            dict_args['errCode'] = "E64002"
            dict_args['param1'] = "Dedy's C1"
            dict_args['param2'] = "Filter's N//16"
            dict_args['actual_value'] = "{}, {}". format(
                self.shape_grads_5hd[1], ceil_div(self.weight_shape[0], 16))
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if ceil_div(self.weight_shape[1], 16) != self.shape_x_5hd[1]:
            dict_args = {}
            dict_args['errCode'] = "E64002"
            dict_args['param1'] = "Fmap's C1"
            dict_args['param2'] = "Filter's C//16"
            dict_args['actual_value'] = "{}, {}". format(
                self.shape_x_5hd[1], ceil_div(self.weight_shape[1], 16))
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if self.shape_grads_5hd[0] != self.shape_x_5hd[0]:
            dict_args = {}
            dict_args['errCode'] = "E64002"
            dict_args['param1'] = "Fmap's N"
            dict_args['param2'] = "Dedy's N"
            dict_args['actual_value'] = "{}, {}".\
                format(self.shape_x_5hd[0], self.shape_grads_5hd[0])
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        # special supporting for a unique case, there are 2 conditions:
        # (1) height & weight of x/output_backprop/filter are all 1
        # (2) strides is [1,1]
        def _is_all_one_case():
            if self.stride == [1, 1] and self.shape_x_5hd[2:4] == [1, 1] \
                and self.shape_grads_5hd[2:4] == [1, 1] \
                and self.weight_shape[2:4] == [1, 1]:
                self.flag_all_one_case = True

        # special supporting for another unique case, there are 2 conditions:
        # (1) weight_height, weight_width in range[1,11]
        # (2) fmap_height/width + pad_top/left + pad_bottom/right is
        # weight_height/width
        def _is_load3d_special_case():
            # limitation by chip:
            # Ascend910 load3d not support
            # when only fmap w after padding equals to filter w
            if pver().is_cloud_version() \
                and fmap_height_after_pad != kernel_height \
                and fmap_width_after_pad == kernel_width:
                self.flag_load3d_special_case = False
            if (1 <= kernel_height <= 11) and (1 <= kernel_width <= 11) \
                and (fmap_height_after_pad == kernel_height
                     or fmap_width_after_pad == kernel_width):
                self.flag_load3d_special_case = True

        # conv1d situation, all params in h is 1
        # support w be in [1,16000]
        def _is_conv1d_situation():
            if fmap_height_after_pad == 1 and kernel_height == 1 \
                and stride_height == 1 and dilationh == 1:
                self.conv1d_situation = True

        _is_all_one_case()
        _is_load3d_special_case()
        _is_conv1d_situation()

        # individual range check
        grads_hw_min = 2
        grads_hw_max = 4096
        fmap_hw_min = 1
        fmap_hw_max = 4096
        if self.flag_load3d_special_case or self.flag_all_one_case:
            grads_hw_min = 1
        if self.conv1d_situation:
            grads_hw_min = 1
            grads_hw_max = CONV1D_MAX_W
            fmap_hw_max = CONV1D_MAX_W

        check_variable_range(grads_height, grads_hw_min, grads_hw_max,
                             "height of out_backprop")
        check_variable_range(grads_width, grads_hw_min, grads_hw_max,
                             "width of out_backprop")
        check_variable_range(fmap_height, fmap_hw_min, fmap_hw_max,
                             "height of x")
        check_variable_range(fmap_width, fmap_hw_min, fmap_hw_max,
                             "width of x")
        check_variable_range(kernel_height, 1, 255, "height of filter")
        check_variable_range(kernel_width, 1, 255, "width of filter")

        check_attr_rule(self.stride, 2, [1, 63],
                        "[strideH, strideW]", "stride")
        check_attr_rule(self.pad, 4, [0, 255], "[up,down,left,right]", "pad")
        return True

    def deconv_dw_input_check_2(self): # pylint: disable=R0914
        """
        do input parameters check part2

        """
        if self.shape_x_5hd[4] != BLOCK_SIZE:
            dict_args = {}
            dict_args['errCode'] = "E60000"
            dict_args['param_name'] = "axis C0 of x"
            dict_args['expected_value'] = BLOCK_SIZE
            dict_args['input_value'] = str(self.shape_x_5hd[4])
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if self.shape_grads_5hd[4] != BLOCK_SIZE:
            dict_args = {}
            dict_args['errCode'] = "E60000"
            dict_args['param_name'] = "axis C0 of out_backprop"
            dict_args['expected_value'] = BLOCK_SIZE
            dict_args['input_value'] = str(self.shape_grads_5hd[4])
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        if self.shape_x_5hd[0] != self.shape_grads_5hd[0]:
            dict_args = {}
            dict_args['errCode'] = "E64002"
            dict_args['param1'] = "input_x's batch'"
            dict_args['param2'] = "out_backprop's batch"
            dict_args['actual_value'] = "{}, {}".format(
                self.shape_x_5hd[0], self.shape_grads_5hd[0])
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        stride_height, stride_width = self.stride
        pad_top, pad_bottom, pad_left, pad_right = self.pad
        _, grads_channel_1, grads_height, grads_width, grads_c0 \
            = self.shape_grads_5hd
        _, fmap_channel_1, fmap_height, fmap_width, fmap_c0 = self.shape_x_5hd
        _, _, kernel_height, kernel_width = self.weight_shape
        dilationn, dilationc, dilationh, dilationw = self.dilation

        def _check_dilation():
            dilation_min = 1
            dilation_max = 255
            if dilationn != 1 or dilationc != 1:
                dict_args = dict()
                dict_args["errCode"] = "E60023"
                dict_args["dilation_n"] = str(dilationn)
                dict_args["dilation_c"] = str(dilationc)
                raise RuntimeError(dict_args,
                                   err_man.get_error_message(dict_args))
            check_variable_range(dilationh, dilation_min, dilation_max,
                                 "dilationh")
            check_variable_range(dilationw, dilation_min, dilation_max,
                                 "dilationw")
        _check_dilation()

        # coupling range check
        dilation_kernel_height = (kernel_height - 1) * dilationh + 1
        dilation_kernel_width = (kernel_width - 1) * dilationw + 1
        computed_grads_height = (fmap_height - dilation_kernel_height + \
                                 pad_top + pad_bottom)//stride_height + 1
        computed_grads_width = (fmap_width - dilation_kernel_width + \
                                 pad_left + pad_right)//stride_width + 1
        if computed_grads_height != grads_height:
            dict_args["errCode"] = "E60024"
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if computed_grads_width != grads_width:
            dict_args["errCode"] = "E60025"
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        fmap_height_after_pad = fmap_height + pad_top + pad_bottom
        fmap_width_after_pad = fmap_width + pad_left + pad_right
        if dilation_kernel_height > fmap_height_after_pad:
            dict_args = dict()
            dict_args["errCode"] = "E60014"
            dict_args["h_of_x"] = str(fmap_height_after_pad)
            dict_args["h_of_filter"] = str(dilation_kernel_height)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        if dilation_kernel_width > fmap_width_after_pad:
            dict_args = dict()
            dict_args["errCode"] = "E60015"
            dict_args["w_of_x"] = str(fmap_width_after_pad)
            dict_args["w_of_filter"] = str(dilation_kernel_width)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))
        if pad_top >= dilation_kernel_height \
            or pad_bottom >= dilation_kernel_height:
            dict_args = dict()
            dict_args["errCode"] = "E64005"
            dict_args["direction"] = 'H'
            dict_args["pads_dir"] = "pad_top and pad_down"
            dict_args["pads_value"] = "[{}, {}]".format(pad_top, pad_bottom)
            dict_args["filter_value"] = str(dilation_kernel_height)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        if pad_left >= dilation_kernel_width \
            or pad_right >= dilation_kernel_width:
            dict_args = dict()
            dict_args["errCode"] = "E64005"
            dict_args["direction"] = 'W'
            dict_args["pads_dir"] = "pad_left and pad_right"
            dict_args["pads_value"] = "[{}, {}]".format(pad_left, pad_right)
            dict_args["filter_value"] = str(dilation_kernel_width)
            raise RuntimeError(dict_args,
                               err_man.get_error_message(dict_args))

        # int64 addressing limit of tvm
        kernel_fractal = (fmap_channel_1*kernel_height*kernel_width,
                          grads_channel_1*grads_c0,
                          fmap_c0)

        check_addressing_rule(self.shape_grads_5hd, 2, DATA_SIZE_LIMIT_INT64,
                              'shape_grads_5hd')
        check_addressing_rule(self.shape_x_5hd, 2, DATA_SIZE_LIMIT_INT64,
                              'shape_x_5hd')
        # because of atomic write, dw_ubuf does not tiled, cannot exceed int32
        # limit (change to int64 limit after tvm v0.6)

        check_addressing_rule(kernel_fractal, 4, DATA_SIZE_LIMIT_INT64,
                              'kernel_fractal')
        return True

    def deconv_dw_access(self):
        """
        complete compute generation, including input check,
        compute definition and result record

        """

        self.deconv_dw_input_check_1()
        self.deconv_dw_input_check_2()
        self.deconv_dw_compute()
        self.res_tensor = self.dw_ddr  # return tensor of this file to topi

    def deconv_dw_compute(self): # pylint: disable=R0914

        """
        complege compute definition

        """

        def _grads_2_fractal(grads_shape_fractal, grads_2_matrix):
            """
            compute definiton of loading grads_matrix to L0A

            Parameters:
            ----------
            grads_shape_fractal : shape of result tensor in L0A

            grads_2_matrix : input tensor in L1

            Returns
            -------
            None
            """

            def __grads_2_fractal_compute(indices, grads_2_matrix):
                """
                do coordinate calculation

                """

                batch_indices, grads_c1_indices, hw_mad_1_indices, \
                grads_c0_indices, hw_mad_0_indices = indices

                batch_size_index = batch_indices
                grads_channel_1_index = grads_c1_indices
                grads_hw_index = hw_mad_1_indices * BLOCK_SIZE \
                                 + hw_mad_0_indices
                grads_c0_index = grads_c0_indices

                return grads_2_matrix(batch_size_index, grads_channel_1_index,
                                      grads_hw_index, grads_c0_index)

            return tvm.compute(grads_shape_fractal,
                               lambda *indices: \
                                   __grads_2_fractal_compute(indices,
                                                             grads_2_matrix),
                               name='grads_2_fractal',
                               tag='grads_2_fractal')

        def _fmap_2_fractal_load2d(fmap_shape_fractal, fmap_2_matrix):
            """
            compute definiton of loading fmap_matrix to L0B

            Parameters:
            ----------
            fmap_shape_fractal : shape of result tensor in L0B

            fmap_2_matrix : input tensor in L1

            Returns
            -------
            None
            """

            def __fmap_2_fractal_load2d_compute(indices, fmap_2_matrix):
                """
                do coordinate calculation

                """
                batch_indices, hw_mad_1_indices, fmap_c1_indices, \
                fmap_c0_indices, hw_mad_0_indices = indices

                batch_size_index = batch_indices
                fmap_channel_1_index = fmap_c1_indices
                fmap_hw_index = hw_mad_1_indices * BLOCK_SIZE \
                                + hw_mad_0_indices
                fmap_c0_index = fmap_c0_indices

                return fmap_2_matrix(batch_size_index, fmap_channel_1_index,
                                     fmap_hw_index, fmap_c0_index)

            return tvm.compute(fmap_shape_fractal,
                               lambda *indices:
                               __fmap_2_fractal_load2d_compute
                               (indices, fmap_2_matrix),
                               name='famp_2_fractal',
                               tag='famp_2_fractal')
        fmap_dtype = self.fmap_dtype

        batch_size, grads_channel_1, grads_height, grads_width, grads_c0 \
            = self.shape_grads_5hd
        _, fmap_channel_1, fmap_height, fmap_width, fmap_c0 = self.shape_x_5hd
        _, _, kernel_height, kernel_width = self.weight_shape

        # align to 16
        hw_mad = (grads_height*grads_width + BLOCK_SIZE - 1) \
                 // BLOCK_SIZE*BLOCK_SIZE

        # move grads to L1
        grads_shape_matrix = (batch_size,
                              grads_channel_1,
                              grads_height*grads_width,
                              grads_c0)
        self.shapelist['grads_matrix'] = grads_shape_matrix

        grads_matrix = self.grads_2_matrix(grads_shape_matrix, self.grads)

        # move grads_matrix to L0A and do transpose
        grads_shape_fractal = (batch_size,
                               grads_channel_1,
                               hw_mad//BLOCK_SIZE,
                               grads_c0,
                               BLOCK_SIZE)

        self.shapelist['grads_fractal'] = grads_shape_fractal
        grads_fractal = _grads_2_fractal(grads_shape_fractal,
                                         grads_matrix)
        if not self.flag_all_one_case:
            # shape of fmap_original_matrix, corresponding to set_fmatrix
            fmap_shape_original_matrix = (batch_size,
                                          grads_height*grads_width,
                                          fmap_channel_1,
                                          kernel_height,
                                          kernel_width,
                                          fmap_c0)


            self.shapelist['fmap_original_matrix'] = fmap_shape_original_matrix

            fmap_matrix = self.fmap_2_matrix(fmap_shape_original_matrix,
                                             self.fmap, fmap_dtype)
            # move fmap to L0B
            fmap_shape_fmap_matrix = (batch_size, hw_mad//BLOCK_SIZE,
                                      fmap_channel_1 * kernel_height *
                                      kernel_width, fmap_c0,
                                      BLOCK_SIZE)
            self.shapelist['fmap_fmap_matrix'] = fmap_shape_fmap_matrix

            fmap_fractal = self.fmap_2_fractal(fmap_shape_fmap_matrix,
                                               fmap_matrix, fmap_dtype)

        # else: all_one_case, using load_2d instead of load_3d
        else:
            # shape of fmap_matrix
            fmap_shape_matrix = (batch_size,
                                 fmap_channel_1,
                                 fmap_height*fmap_width,
                                 fmap_c0)

            self.shapelist['fmap_matrix'] = fmap_shape_matrix

            fmap_matrix = self.fmap_2_matrix_load2d(fmap_shape_matrix,
                                                    self.fmap)

            # move fmap to L0B
            fmap_shape_fractal = (batch_size,
                                  hw_mad//BLOCK_SIZE,
                                  fmap_channel_1*kernel_height*kernel_width,
                                  fmap_c0,
                                  BLOCK_SIZE)
            self.shapelist['fmap_matrix'] = fmap_shape_fractal

            fmap_fractal = _fmap_2_fractal_load2d(fmap_shape_fractal,
                                                  fmap_matrix)

        # shape of result dw [n1,m,n0]
        dw_shape = (fmap_channel_1*kernel_height*kernel_width,
                    grads_channel_1*grads_c0, fmap_c0)
        self.shapelist['dw'] = dw_shape

        # do mmad
        dw_cc = self.mad(dw_shape, grads_fractal, fmap_fractal)

        def _lambda_func_dw_cc(*params):
            return dw_cc(*params)

        # move dw_cc to UB
        dw_ubuf = tvm.compute(dw_shape, _lambda_func_dw_cc,
                              name='dw_ubuf', tag="dw_ubuf")

        def _lambda_func_dw_ubuf(*params):
            return dw_ubuf(*params)

        # move to ddr
        self.dw_ddr = tvm.compute(dw_shape, _lambda_func_dw_ubuf,
                                  name='dw_ddr', tag=self.optag + "dw_ddr",
                                  attrs={'kernel_name': self.kernel_name})

        return 1

    def grads_2_matrix(self, grads_shape_matrix, grads):
        """
        compute definiton of loading grads to L1

        Parameters:
        ----------
        grads_shape_matrix : shape of result tensor in L1

        grads : input tensor in ddr

        Returns
        -------
        None
        """
        def __grads_2_matrix_compute(indices, grads):
            """
            do coordinate calculation

            """

            grads_width = self.shapelist['grads_5hd'][3]

            batch_indices, grads_c1_indices, hw_mad_indices, grads_c0_indices \
            = indices

            # calculate index of grads according to indice of grads_matrix
            batch_size_index = batch_indices
            grads_channel_1_index = grads_c1_indices
            grads_height_index = (hw_mad_indices // grads_width)
            grads_width_index = (hw_mad_indices % grads_width)
            grads_c0_index = grads_c0_indices

            return grads(batch_size_index, grads_channel_1_index,
                         grads_height_index, grads_width_index, grads_c0_index)

        return tvm.compute(grads_shape_matrix,
                           lambda *indices:
                           __grads_2_matrix_compute(indices, grads),
                           name='grads_2_matrix',
                           tag='grads_2_matrix')

    def fmap_2_matrix(self, fmap_shape_original_matrix, fmap, fmap_dtype):
        """
        compute definiton of set_fmatrix

        Parameters:
        ----------
        fmap_shape_original_matrix : shape of result tensor in L1
        in shape (batch_size,
                  grads_height*grads_width,
                  fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap : input tensor in L1

        fmap_dtype : data type of fmap
        in shape (batch_size, fmap_channel_1, fmap_height, fmap_width, C0)

        Returns
        -------
        None
        """

        def __fmap_2_matrix_compute(indices,  # pylint:disable=R0913,R0914
                                    fmap, kernel_width,
                                    pad_left=0, pad_right=0, pad_top=0,
                                    strideh=1, stridew=1, dilationh=1,
                                    dilationw=1):
            """
            do coordinate calculation

            """

            _, _, fmap_height, fmap_width, _ = self.shapelist['fmap_5hd']

            batch_indices, \
            hw_fuse_indices, \
            fmap_c1_indices, \
            kernel_height_indices, \
            kernel_width_indices, \
            fmap_c0_indices = indices

            dilation_kernel_width = kernel_width \
                                    + (kernel_width - 1) * (dilationw - 1)
            fmap_width_after_pad = fmap_width + pad_left + pad_right
            width_out = (fmap_width_after_pad - dilation_kernel_width) \
                        // stridew + 1

            n_index = batch_indices
            c1_index = fmap_c1_indices
            h_index = (hw_fuse_indices // width_out) * strideh \
                      + kernel_height_indices * dilationh
            w_index = (hw_fuse_indices % width_out) * stridew \
                      + kernel_width_indices * dilationw
            c0_index = fmap_c0_indices

            # if index belongs to padding and 16 align, select 0
            return tvm.select(tvm.any(h_index < pad_top,
                                      h_index > fmap_height + pad_top - 1,
                                      w_index < pad_left,
                                      w_index > fmap_width + pad_left - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap(n_index, c1_index, h_index-pad_top, \
                                   w_index-pad_left, c0_index))

        pad_top, _, pad_left, pad_right = self.pad
        strideh, stridew = self.stride
        _, _, dilationh, dilationw = self.dilation

        kernel_width = fmap_shape_original_matrix[4]
        return tvm.compute(fmap_shape_original_matrix,
                           lambda *indices:
                           __fmap_2_matrix_compute(indices, fmap, kernel_width,
                                                   pad_left=pad_left,
                                                   pad_right=pad_right,
                                                   pad_top=pad_top,
                                                   strideh=strideh,
                                                   stridew=stridew,
                                                   dilationh=dilationh,
                                                   dilationw=dilationw),
                           name='fmap_2_col_matrix',
                           tag='fmap_2_col_matrix',
                           attrs={'pad': self.pad, 'stride': self.stride,
                                  'dilation': self.dilation,
                                  'kernel_size': self.weight_shape})

    def fmap_2_matrix_load2d(self, fmap_shape_matrix, fmap):
        """
        compute definiton of loading fmap to L1

        Parameters:
        ----------
        fmap_shape_matrix : shape of result tensor in L1

        fmap : input tensor in ddr

        Returns
        -------
        None
        """
        def __fmap_2_matrix_load2d_compute(indices, fmap):
            """
            do coordinate calculation

            """
            fmap_width = self.shapelist['fmap_5hd'][3]
            batch_indices, fmap_c1_indices, hw_mad_indices, fmap_c0_indices \
            = indices
            batch_size_index = batch_indices
            fmap_channel_1_index = fmap_c1_indices
            fmap_height_index = (hw_mad_indices // fmap_width)
            fmap_width_index = (hw_mad_indices % fmap_width)
            fmap_c0_index = fmap_c0_indices
            return fmap(batch_size_index, fmap_channel_1_index,
                        fmap_height_index, fmap_width_index, fmap_c0_index)
        return tvm.compute(fmap_shape_matrix,
                           lambda *indices:
                           __fmap_2_matrix_load2d_compute(indices, fmap),
                           name='fmap_2_matrix',
                           tag='fmap_2_matrix',
                           attrs={'pad': self.pad, 'stride': self.stride,
                                  'dilation': self.dilation,
                                  'kernel_size': self.weight_shape})

    def fmap_2_fractal(self, fmap_shape_fmap_matrix,
                       fmap_2_col_matrix, fmap_dtype):
        """
        compute definiton of loading fmap to L0B

        Parameters:
        ----------
        fmap_shape_fmap_matrix : shape of result tensor in L0B
        in shape (batch_size,
                  hw_mad//block_size_K,
                  fmap_channel_1*kernel_height*kernel_width,
                  fmap_c0,
                  block_size_K)

        fmap_2_col_matrix : input tensor in L1
        in shape (batch_size,
                  grads_height*grads_width,
                  fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap_dtype : data type of fmap_2_col_matrix


        Returns
        -------
        None
        """

        def __fmap_2_fractal_compute(  # pylint:disable=R0913,R0914
                indices, fmap_2_col_matrix):
            """
            do coordinate calculation

            """

            _, hw_fuse, _, kernel_height, kernel_width, _ \
                = self.shapelist['fmap_original_matrix']

            # batch_size
            # hw_mad//block_size_K
            # fmap_channel_1*kernel_height*kernel_width
            # fmap_c0
            # block_size_K
            n_vm_index, hw_mad_1_indices, fkk_indices, \
            fmap_c0_indices, hw_mad_0_indices \
                = indices

            hw_vm_index = hw_mad_1_indices*BLOCK_SIZE + hw_mad_0_indices
            c1_vm_index = (((fkk_indices*BLOCK_SIZE + fmap_c0_indices)
                            // BLOCK_SIZE) // kernel_width) // kernel_height
            kh_vm_index = (((fkk_indices*BLOCK_SIZE + fmap_c0_indices)
                            // BLOCK_SIZE) // kernel_width) % kernel_height
            kw_vm_index = ((fkk_indices*BLOCK_SIZE + fmap_c0_indices)
                           // BLOCK_SIZE) % kernel_width
            c0_vm_index = \
                (fkk_indices*BLOCK_SIZE + fmap_c0_indices) % BLOCK_SIZE

            # select padding and 16 align
            return tvm.select(tvm.any(hw_vm_index < 0,
                                      hw_vm_index > hw_fuse - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap_2_col_matrix(n_vm_index, hw_vm_index,
                                                c1_vm_index, kh_vm_index,
                                                kw_vm_index, c0_vm_index))

        return tvm.compute(fmap_shape_fmap_matrix,
                           lambda *indices:
                           __fmap_2_fractal_compute(indices,
                                                    fmap_2_col_matrix),
                           name='fmap_2_col_fractal',
                           tag='fmap_2_col_fractal')

    def mad(self, mad_shape, grads, fmap):  # pylint: disable=R0913,R0914
        """
        calculate mad result tensor
        Parameters
        ----------
        mad_shape : result shape
        (fmap_channel_1*kernel_height*kernel_width, grads_channel, fmap_c0)

        grads : tensor in L0A
        grads_shape_fractal = (batch_size,
                               grads_channel_1,
                               hw_mad//block_size_K,
                               grads_c0,
                               block_size_K)

        fmap : tensor in L0B
        fmap_shape_fmap_matrix = (batch_size,
                                  hw_mad//block_size_K,
                                  fmap_channel_1*kernel_height*kernel_width,
                                  fmap_c0,
                                  block_size_K)

        Returns
        -------
        None
        """

        batch_size, _, grads_height, grads_width, _ \
            = self.shapelist['grads_5hd']
        hw_fuse = grads_height*grads_width

        batch_axis = tvm.reduce_axis((0, batch_size), name='axis_b')
        k_axis = tvm.reduce_axis((0, hw_fuse), name='axis_k')

        k_1 = k_axis.var // 16
        k_0 = k_axis.var % 16

        mode = "f162f32"
        if self.res_dtype == "float16":
            mode = "f162f16"

        c_col = tvm.compute(mad_shape,
                            lambda fkk, grads_c, fmap_c0:
                            tvm.sum((grads[batch_axis, grads_c // 16,
                                           k_1, grads_c % 16, k_0] *
                                     fmap[batch_axis, k_1, fkk, fmap_c0,
                                          k_0]).astype(self.res_dtype),
                                    axis=[batch_axis, k_axis]),
                            name='dw', tag="dw",
                            attrs={'mode': mode})
        return c_col


@util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (list, tuple),
                       (list, tuple), (list, tuple),
                       (list, tuple), str, str)
def conv2d_backprop_filter_compute(input_x,  # pylint: disable=R0913
                                   out_backprop, filter_sizes,
                                   strides=None, padding=None,
                                   dilations=None, res_dtype="float32",
                                   kernel_name="conv2d_backprop_filter_cce"):
    """
    the DSL interface of conv2d backprop filter compute

    Parameters:
    ----------
    x : the featuremap data, tvm.placeholder, 5HD shape

    out_backprop : the grads data, tvm.placeholder, 5HD shape

    filter_sizes : 4-D shape, specifies the filter sizes

    strides : 2-D shape, specifies in height and width dimension

    padding : 4-D shape, specifies in up/down/left/right dimension

    dilations : 4-D shape, specifies in batch/channel/height/width dimension

    res_dtype : the output data type

    Returns
    -------
    result tensor of conv2d_backprop_filter compute
    """
    if not strides:
        strides = [1, 1]
    if not padding:
        padding = [0, 0, 0, 0]
    if not dilations:
        dilations = [1, 1, 1, 1]
    deconv_dw_object = Conv2dBackpropFilter(input_x, out_backprop,
                                            filter_sizes,
                                            strides=strides,
                                            padding=padding,
                                            dilations=dilations,
                                            res_dtype=res_dtype,
                                            kernel_name=kernel_name)
    deconv_dw_object.deconv_dw_access()

    return deconv_dw_object.res_tensor
