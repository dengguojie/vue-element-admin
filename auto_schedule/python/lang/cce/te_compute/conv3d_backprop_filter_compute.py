# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
conv3d backprop filter DSL interface.
"""
import functools

import te.platform as tbe_platform
from te.lang.cce.te_compute import util as te_util
from te.utils import para_check
from te.utils.error_manager import error_manager_util
from te import tvm


# fractal size, only support 16 for now
_BLOCK_SIZE = 16
# maximum of int64 (2**63 - 1)
_DATA_SIZE_LIMIT_INT64 = 9223372036864776807


def _check_shape_rule(shape, dim, formats, name):
    """
    check shape

    """
    if len(shape) != dim:
        args_dict = {
            'errCode': 'E60006',
            'param_name': name,
            'expected_length': str(dim),
            'length': str(len(shape))
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))
    for dim_x in shape:
        if (not isinstance(dim_x, int)) or dim_x <= 0:
            args_dict = {
                'errCode': 'E62509',
                'param_name': name
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))


def _check_attr_rule(attr, dim, attr_limit, formats, name):
    """
    check attribute

    """
    attr_min = attr_limit[0]
    attr_max = attr_limit[1]
    if len(attr) != dim:
        args_dict = {
            'errCode': 'E60006',
            'param_name': name,
            'expected_length': str(dim),
            'length': str(len(attr))
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))
    for attr_x in attr:
        if (not isinstance(attr_x, int)) or attr_x < attr_min or attr_x > attr_max:
            args_dict = {
                'errCode': 'E60011',
                'range': '[{},{}]'.format(attr_min, attr_max),
                'attr_name': name,
                'value': str(attr_x)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))


def _check_variable_range(variable, minimum, maximum, name):
    """
    check variable range

    """
    if variable < minimum or variable > maximum:
        args_dict = {
            'errCode': 'E60011',
            'range': '[{},{}]'.format(minimum, maximum),
            'attr_name': name,
            'value': str(variable)
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))


def _check_addressing_rule(shape, byte_count, limit):
    """
    check addressing limit

    """
    # product of all dimension
    product = functools.reduce(lambda x, y: x * y, shape[:])
    if product * byte_count > limit:
        args_dict = {
            'errCode': 'E60011',
            'range': '(,{}]'.format(limit),
            'attr_name': 'address_byte',
            'value': str(product * byte_count)
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))


class Conv3dBackpropFilter:
    """
    Conv3dBackpropFilter: compute definition of conv3d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    _deconv_dw_input_check_1: parameters check part 1

    _deconv_dw_input_check_2: parameters check part 2

    _deconv_dw_access: compute generation

    _deconv_dw_compute: compute process

    _grads_2_matrix: compute definition of loading grads to cbuf

    _fmap_2_matrix: compute definiton of set_fmatrix

    _fmap_2_matrix_load2d: compute definiton of loading fmap to L1

    _fmap_2_fractal: compute definiton of loading fmap to L0B

    _mad: compute definition of mmad

    """

    def __init__(self,
                 input_x,
                 out_backprop,
                 filter_sizes,
                 strides,
                 padding,
                 group_dict,
                 dilations,
                 res_dtype="float32",
                 kernel_name="conv3d_backprop_filter_cce"):
        """
        initialization

        Parameters:
        ----------
        input_x : the featuremap data, tvm.placeholder, 6hd shape
        [N, D, C1, H, W, C0]

        out_backprop : the grads data, tvm.placeholder, 6hd shape
        [N, DO, GRADS_C1, HO, WO, GRADS_C0]

        filter_sizes : 5-D shape, specifies the filter sizes
        [GRADS_C, KD, C, KH, KW]

        strides : 3-D shape in depth, height and width dimension
        [STRIDE_D, STRIDE_H, STRIDE_W]

        padding : 6-D shape in front/back/up/down/left/right dimension
        [PAD_D, PAD_D, PAD_H, PAD_H, PAD_W, PAD_W]

        dilations : 5-D shape in batch/channel/depth/height/width dimension

        res_dtype : the output data type

        Returns
        -------
        None
        """

        self.shape_list = {}

        self.fmap = input_x
        self.grads = out_backprop

        self.weight_shape = list(filter_sizes)

        self.fmap_dtype = input_x.dtype
        self.grads_dtype = out_backprop.dtype
        self.res_dtype = res_dtype

        self.pad = list(padding)
        self.stride = list(strides)
        self.dilation = list(dilations)
        self.group_dict = group_dict
        self.op_tag = "conv3d_backprop_filter"

        self._kernel_name = kernel_name

        # 6hd shape
        # [N, DO, GRADS_C1, HO, WO, GRADS_C0]
        self.shape_grads_6hd = list(x.value for x in self.grads.shape)
        # [N, D, C1, H, W, C0]
        self.shape_x_6hd = list(x.value for x in self.fmap.shape)

        self.shape_list['grads_6hd'] = self.shape_grads_6hd
        self.shape_list['fmap_6hd'] = self.shape_x_6hd

        # flag of special case
        self.flag_all_one_case = False
        self.dw_ddr = []
        self.res_tensor = self.dw_ddr  # return tensor of this file to topi

        # special supporting for a unique case, there are 2 conditions:
        # (1) height & weight of x/output_backprop/filter are all 1
        # (2) strides is [1,1]
        if (self.stride[1:] == [1, 1] and self.shape_x_6hd[3:5] == [1, 1]
            and self.shape_grads_6hd[3:5] == [1, 1]
            and self.weight_shape[3:5] == [1, 1]):
            self.flag_all_one_case = True

        self.flag_load3d_special_case = False
        # special supporting for another unique case, there are 2 conditions:
        # (1) weight_height, weight_width in range[1,11]
        # (2) fmap_height/width + pad_top/left + pad_bottom/right is
        # weight_height/width
        _, fmap_depth, _, fmap_height, fmap_width, _ = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = self.pad
        _, dilation_d, _, dilation_h, dilation_w = dilations

        filter_dilated_d = (kernel_depth - 1) * dilation_d + 1
        filter_dilated_h = (kernel_height - 1) * dilation_h + 1
        filter_dilated_w = (kernel_width - 1) * dilation_w + 1
        if ((1 <= kernel_height <= 11) and (1 <= kernel_width <= 11)
            and (fmap_height + pad_top + pad_bottom == filter_dilated_h
            or fmap_width + pad_left + pad_right == filter_dilated_w or
            fmap_depth + pad_front + pad_back == filter_dilated_d)):
            self.flag_load3d_special_case = True

    def _deconv_dw_input_check_1(self):
        """
        do input parameters check part1

        """
        # check of data type
        if self.fmap_dtype != "float16":
            args_dict = {
                'errCode': 'E60005',
                'param_name': 'fmap_dtype',
                'expected_dtype_list': 'float16',
                'dtype': str(self.fmap_dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        if self.grads_dtype != "float16":
            args_dict = {
                'errCode': 'E60005',
                'param_name': 'grads_dtype',
                'expected_dtype_list': 'float16',
                'dtype': str(self.grads_dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        is_lhisi_version = tbe_platform.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",
                                                                        "Hi3796CV300CS",
                                                                        "SD3403")
        if is_lhisi_version and self.res_dtype != "float16":
            args_dict = {
                'errCode': 'E60005',
                'param_name': 'res_dtype_lhisi',
                'expected_dtype_list': 'float16',
                'dtype': str(self.res_dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        if not is_lhisi_version and self.res_dtype != "float32":
            args_dict = {
                'errCode': 'E60005',
                'param_name': 'res_dtype',
                'expected_dtype_list': 'float32',
                'dtype': str(self.res_dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        # check shape
        # each element must be positive int
        _check_shape_rule(self.shape_x_6hd, 6, "NDC1HWC0", "x")
        _check_shape_rule(self.shape_grads_6hd, 6, "NDC1HWC0", "out_backprop")
        _check_shape_rule(self.weight_shape, 5, "NDCHW", "filter_sizes")

        _, grads_depth, _, grads_height, grads_width, _ = self.shape_grads_6hd
        _, fmap_depth, _, fmap_height, fmap_width, _ = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape

        if te_util.int_ceil_div(self.weight_shape[0], 16) != self.shape_grads_6hd[2]:
            args_dict = {
                'errCode': 'E62504',
                'backprop_C': str(self.shape_grads_6hd[2]),
                'forward_shape': str(te_util.int_ceil_div(self.weight_shape[0], 16))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        # individual range check
        if not self.flag_all_one_case:
            if not self.flag_load3d_special_case:
                _check_variable_range(grads_depth, 2, 4096,
                                     "depth of out_backprop")
                _check_variable_range(grads_height, 2, 4096,
                                     "height of out_backprop")
                _check_variable_range(grads_width, 2, 4096,
                                     "width of out_backprop")
            else:
                _check_variable_range(grads_depth, 1, 4096,
                                     "depth of out_backprop")
                _check_variable_range(grads_height, 1, 4096,
                                     "height of out_backprop")
                _check_variable_range(grads_width, 1, 4096,
                                     "width of out_backprop")
        _check_variable_range(fmap_depth, 1, 4096, "depth of x")
        _check_variable_range(fmap_height, 1, 4096, "height of x")
        _check_variable_range(fmap_width, 1, 4096, "width of x")
        _check_variable_range(kernel_depth, 1, 256, "depth of filter")
        _check_variable_range(kernel_height, 1, 256, "height of filter")
        _check_variable_range(kernel_width, 1, 256, "width of filter")

        _check_attr_rule(self.stride, 3, [1, 63],
                         "[strideD, strideH, strideW]", "stride")
        _check_attr_rule(self.pad, 6, [0, 255],
                        "[front,back,up,down,left,right]", "pad")
        return True

    def _deconv_dw_input_check_2(self):
        """
        do input parameters check part2

        """
        stride_depth, stride_height, stride_width = self.stride
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = self.pad
        _, grads_depth, grads_channel_1, grads_height, grads_width, grads_c0 = self.shape_grads_6hd
        _, fmap_depth, fmap_channel_1, fmap_height, fmap_width, fmap_c0 = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape
        dilationn, dilationd, dilationc, dilationh, dilationw = self.dilation

        def _check_dilation():
            dilation_min = 1
            dilation_max = 255
            if dilationn != 1 or dilationc != 1:
                args_dict = {
                    'errCode': 'E62510'
                }
                raise RuntimeError(args_dict,
                    error_manager_util.get_error_message(args_dict))
            dilation_dhw = [dilationd, dilationh, dilationw]
            for item in dilation_dhw:
                if item < dilation_min or item > dilation_max:
                    args_dict = {
                        'errCode': 'E60011',
                        'range': '[{},{}]'.format(dilation_min, dilation_max),
                        'attr_name': 'dilation_dhw',
                        'value': str(item)
                    }
                    raise RuntimeError(args_dict,
                        error_manager_util.get_error_message(args_dict))

        _check_dilation()

        if self.shape_x_6hd[5] != _BLOCK_SIZE:
            args_dict = {
                'errCode': 'E62511',
                'shape_c0': str(self.shape_x_6hd[5])
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if self.shape_grads_6hd[5] != _BLOCK_SIZE:
            args_dict = {
                'errCode': 'E62511',
                'shape_c0': str(self.shape_grads_6hd[5])
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        # batch_size should be same
        if self.shape_x_6hd[0] != self.shape_grads_6hd[0]:
            args_dict = {
                'errCode': 'E62503',
                'backprop_N': str(self.shape_grads_6hd[0]),
                'forward_shape': str(self.shape_x_6hd[0])
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        # coupling range check
        dilation_kernel_depth = (kernel_depth - 1) * dilationd + 1
        computed_grads_depth = (fmap_depth - dilation_kernel_depth +
                                pad_front + pad_back)//stride_depth + 1
        if computed_grads_depth != grads_depth:
            args_dict = {
                'errCode': 'E60000',
                'param_name': 'grads_depth',
                'expected_value': str(grads_depth),
                'input_value': str(computed_grads_depth)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        dilation_kernel_height = (kernel_height - 1) * dilationh + 1
        computed_grads_height = (fmap_height - dilation_kernel_height +
                                 pad_top + pad_bottom)//stride_height + 1
        if computed_grads_height != grads_height:
            args_dict = {
                'errCode': 'E60000',
                'param_name': 'grads_height',
                'expected_value': str(grads_height),
                'input_value': str(computed_grads_height)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        dilation_kernel_width = (kernel_width - 1) * dilationw + 1
        computed_grads_width = (fmap_width - dilation_kernel_width +
                                pad_left + pad_right)//stride_width + 1
        if computed_grads_width != grads_width:
            args_dict = {
                'errCode': 'E60000',
                'param_name': 'grads_width',
                'expected_value': str(grads_width),
                'input_value': str(computed_grads_width)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        fmap_depth_after_pad = fmap_depth + pad_front + pad_back
        fmap_height_after_pad = fmap_height + pad_top + pad_bottom
        fmap_width_after_pad = fmap_width + pad_left + pad_right
        if (dilation_kernel_height > fmap_height_after_pad
            or dilation_kernel_width > fmap_width_after_pad
            or dilation_kernel_depth > fmap_depth_after_pad):
            args_dict = {
                'errCode': 'E60108',
                'reason': 'depth/height/width of filter '
                          'cannot exceed that of x.'
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        def _check_pad():
            if pad_front >= dilation_kernel_depth or pad_back >= dilation_kernel_depth:
                args_dict = {
                    'errCode': 'E60108',
                    'reason': 'pad in front/back should '
                              'less than depth of filter.'
                }
                raise RuntimeError(args_dict,
                    error_manager_util.get_error_message(args_dict))

            if pad_top >= dilation_kernel_height or pad_bottom >= dilation_kernel_height:
                args_dict = {
                    'errCode': 'E60108',
                    'reason': 'pad in up/down should '
                              'less than height of filter.'
                }
                raise RuntimeError(args_dict,
                    error_manager_util.get_error_message(args_dict))

            if pad_left >= dilation_kernel_width or pad_right >= dilation_kernel_width:
                args_dict = {
                    'errCode': 'E60108',
                    'reason': 'pad in left/right should '
                              'less than width of filter.'
                }
                raise RuntimeError(args_dict,
                    error_manager_util.get_error_message(args_dict))

        _check_pad()

        # int64 addressing limit of tvm
        kernel_fractal = (kernel_depth * fmap_channel_1 * kernel_height *
                          kernel_width, grads_channel_1 * grads_c0, fmap_c0)

        _check_addressing_rule(self.shape_grads_6hd, 2, _DATA_SIZE_LIMIT_INT64)
        _check_addressing_rule(self.shape_x_6hd, 2, _DATA_SIZE_LIMIT_INT64)
        _check_addressing_rule(kernel_fractal, 4, _DATA_SIZE_LIMIT_INT64)
        return True

    def _deconv_dw_access(self):
        """
        complete compute generation, including input check,
        compute definition and result record

        """

        self._deconv_dw_input_check_1()
        self._deconv_dw_input_check_2()
        self._deconv_dw_compute()
        self.res_tensor = self.dw_ddr

    def _deconv_dw_compute(self):
        """
        complete compute definition

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
                group_dict = self.group_dict
                (group_indices, batch_indices, grads_c1_indices,
                 hw_mad_1_indices, grads_c0_indices, hw_mad_0_indices) = indices

                batch_size_index = batch_indices
                grads_channel_1_index = (
                    group_indices *
                    (group_dict['cout_g'] // 16) + grads_c1_indices)
                grads_hw_index = hw_mad_1_indices * _BLOCK_SIZE + hw_mad_0_indices
                grads_c0_index = grads_c0_indices

                return grads_2_matrix(batch_size_index, grads_channel_1_index,
                                      grads_hw_index, grads_c0_index)

            return tvm.compute(grads_shape_fractal,
                               lambda *indices:
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
                (group_indices, batch_indices,
                 hw_mad_1_indices, fmap_c1_indices,
                 fmap_c0_indices, hw_mad_0_indices) = indices

                batch_size_index = batch_indices
                depth_index = fmap_c1_indices // self.group_dict['cin1_g']
                cin_index = fmap_c1_indices % self.group_dict['cin1_g']
                c1_index = (depth_index * self.shape_x_6hd[2] + group_indices *
                            self.group_dict['cin1_g'] + cin_index)
                fmap_hw_index = hw_mad_1_indices * _BLOCK_SIZE + hw_mad_0_indices
                fmap_c0_index = fmap_c0_indices

                return fmap_2_matrix(batch_size_index, c1_index,
                                     fmap_hw_index, fmap_c0_index)

            return tvm.compute(
                fmap_shape_fractal,
                lambda *indices: __fmap_2_fractal_load2d_compute(
                    indices, fmap_2_matrix),
                name='famp_2_fractal',
                tag='famp_2_fractal')

        fmap_dtype = self.fmap_dtype

        (batch_size, grads_depth, grads_channel_1, grads_height, grads_width,
         grads_c0) = self.shape_grads_6hd
        (_, fmap_depth, fmap_channel_1, fmap_height, fmap_width,
         fmap_c0) = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape
        real_g = self.group_dict['real_g']
        fmap_channel1_g = self.group_dict['cin1_g']
        grads_channel_g = self.group_dict['cout_g']
        cin_ori = self.group_dict['cin_ori']
        cout_ori = self.group_dict['cout_ori']
        # align to 16
        hw_ori = grads_height * grads_width
        hw_mad_1 = (hw_ori + _BLOCK_SIZE - 1) // _BLOCK_SIZE

        # move grads to L1
        grads_shape_matrix = (batch_size * grads_depth, grads_channel_1,
                              hw_ori, grads_c0)
        self.shape_list['grads_matrix'] = grads_shape_matrix

        grads_matrix = self._grads_2_matrix(grads_shape_matrix, self.grads)

        # move grads_matrix to L0A and do transpose
        grads_shape_fractal = (real_g, batch_size * grads_depth,
                               grads_channel_g // grads_c0,
                               hw_mad_1, grads_c0, _BLOCK_SIZE)

        self.shape_list['grads_fractal'] = grads_shape_fractal
        grads_fractal = _grads_2_fractal(grads_shape_fractal,
                                         grads_matrix)

        stride_depth, _, _ = self.stride
        pad_front, _, _, _, _, _ = self.pad

        def _tensor_to_al1():
            fmap_al1_shape = (batch_size * grads_depth,
                              kernel_depth * fmap_channel_1, fmap_height,
                              fmap_width, fmap_c0)
            fmap_l1 = tvm.compute(
                fmap_al1_shape,
                lambda i, j, k, l, m: tvm.select(
                    tvm.all((i % grads_depth) * stride_depth + j //
                            fmap_channel_1 >= pad_front,
                            (i % grads_depth) * stride_depth + j //
                            fmap_channel_1 < pad_front + fmap_depth),
                    self.fmap(
                        i // grads_depth, i % grads_depth * stride_depth + j //
                        fmap_channel_1 - pad_front, j % fmap_channel_1, k, l, m
                    )),
                name='fmap_l1',
                tag='fmap_l1')
            return fmap_l1

        if not self.flag_all_one_case:
            fmap_l1 = _tensor_to_al1()
            # shape of fmap_original_matrix, corresponding to set_fmatrix
            fmap_shape_original_matrix = (batch_size * grads_depth,
                                          hw_ori,
                                          kernel_depth * fmap_channel_1,
                                          kernel_height, kernel_width, fmap_c0)

            self.shape_list['fmap_original_matrix'] = fmap_shape_original_matrix

            fmap_matrix = self._fmap_2_matrix(fmap_shape_original_matrix,
                                              fmap_l1, fmap_dtype)
            # move fmap to L0B
            fmap_shape_fmap_matrix = (real_g, batch_size * grads_depth,
                                      hw_mad_1, kernel_depth *
                                      fmap_channel1_g * kernel_height *
                                      kernel_width, fmap_c0, _BLOCK_SIZE)
            self.shape_list['fmap_fmap_matrix'] = fmap_shape_fmap_matrix

            fmap_fractal = self._fmap_2_fractal(fmap_shape_fmap_matrix,
                                                fmap_matrix, fmap_dtype)

        # else: all_one_case, using load_2d instead of load_3d
        else:
            # shape of fmap_matrix
            fmap_shape_matrix = (batch_size * grads_depth,
                                 kernel_depth * fmap_channel_1,
                                 fmap_height * fmap_width, fmap_c0)

            self.shape_list['fmap_matrix'] = fmap_shape_matrix

            fmap_matrix = self._fmap_2_matrix_load2d(fmap_shape_matrix,
                                                     self.fmap)
            # move fmap to L0B
            fmap_shape_fractal = (real_g, batch_size * grads_depth, hw_mad_1,
                                  kernel_depth * fmap_channel1_g *
                                  kernel_height * kernel_width, fmap_c0,
                                  _BLOCK_SIZE)
            self.shape_list['fmap_fractal'] = fmap_shape_fractal

            fmap_fractal = _fmap_2_fractal_load2d(fmap_shape_fractal,
                                                  fmap_matrix)
        # shape of result dw [n1,m,n0]
        dw_shape = (real_g, kernel_depth * fmap_channel1_g * kernel_height *
                    kernel_width, grads_channel_g, fmap_c0)
        self.shape_list['dw'] = dw_shape

        # do mmad
        dw_cc = self._mad(dw_shape, grads_fractal, fmap_fractal)

        def _lambda_func_dw_cc(*params):
            return dw_cc(*params)

        # move dw_cc to UB
        dw_ubuf = tvm.compute(dw_shape,
                              _lambda_func_dw_cc,
                              name='dw_ubuf',
                              tag="dw_ubuf")

        def _lambda_func_dw_ubuf(*params):
            return dw_ubuf(*params)

        # move to ddr
        self.dw_ddr = tvm.compute(dw_shape,
                                  _lambda_func_dw_ubuf,
                                  name='dw_ddr',
                                  tag=self.op_tag + "dw_ddr")

        return 1

    def _grads_2_matrix(self, grads_shape_matrix, grads):
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
            grads_depth = self.shape_list['grads_6hd'][1]
            grads_width = self.shape_list['grads_6hd'][4]

            batch_indices, grads_c1_indices, hw_indices, grads_c0_indices \
                = indices

            # calculate index of grads according to indice of grads_matrix
            batch_size_index = batch_indices // grads_depth
            grads_depth_index = batch_indices % grads_depth
            grads_channel_1_index = grads_c1_indices
            grads_height_index = (hw_indices // grads_width)
            grads_width_index = (hw_indices % grads_width)
            grads_c0_index = grads_c0_indices

            return grads(batch_size_index, grads_depth_index,
                         grads_channel_1_index, grads_height_index,
                         grads_width_index, grads_c0_index)

        return tvm.compute(
            grads_shape_matrix,
            lambda *indices: __grads_2_matrix_compute(indices, grads),
            name='grads_2_matrix',
            tag='grads_2_matrix')

    def _fmap_2_matrix(self, fmap_shape_original_matrix, fmap, fmap_dtype):
        """
        compute definiton of set_fmatrix

        Parameters:
        ----------
        fmap_shape_original_matrix : shape of result tensor in L1
        in shape (batch_size*grads_depth,
                  grads_height*grads_width,
                  kernel_depth*fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap : input tensor in L1

        fmap_dtype : data type of fmap
        in shape (batch_size, fmap_depth, fmap_channel_1,
                  fmap_height, fmap_width, C0)

        Returns
        -------
        None
        """
        def __fmap_2_matrix_compute(indices,
                                    fmap,
                                    kernel_width,
                                    pad_left=0,
                                    pad_right=0,
                                    pad_top=0,
                                    strideh=1,
                                    stridew=1,
                                    dilationh=1,
                                    dilationw=1):
            """
            do coordinate calculation

            """

            _, _, _, fmap_height, fmap_width, _ = self.shape_list['fmap_6hd']

            (batch_indices, hw_fuse_indices, fmap_c1_indices, kernel_height_indices,
             kernel_width_indices, fmap_c0_indices) = indices

            dilation_kernel_width = kernel_width + (kernel_width - 1) * (dilationw - 1)
            fmap_width_after_pad = fmap_width + pad_left + pad_right
            width_out = (fmap_width_after_pad - dilation_kernel_width) // stridew + 1

            n_index = batch_indices
            c1_index = fmap_c1_indices
            h_index = (hw_fuse_indices // width_out) * strideh + kernel_height_indices * dilationh
            w_index = (hw_fuse_indices % width_out) * stridew + kernel_width_indices * dilationw
            c0_index = fmap_c0_indices
            # if index belongs to padding and 16 align, select 0
            return tvm.select(tvm.any(h_index < pad_top,
                                      h_index > fmap_height + pad_top - 1,
                                      w_index < pad_left,
                                      w_index > fmap_width + pad_left - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap(n_index, c1_index, h_index - pad_top,
                                   w_index - pad_left, c0_index))

        _, _, pad_top, _, pad_left, pad_right = self.pad
        _, strideh, stridew = self.stride
        _, _, _, dilationh, dilationw = self.dilation

        kernel_width = fmap_shape_original_matrix[4]
        return tvm.compute(
            fmap_shape_original_matrix,
            lambda *indices: __fmap_2_matrix_compute(indices,
                                                     fmap,
                                                     kernel_width,
                                                     pad_left=pad_left,
                                                     pad_right=pad_right,
                                                     pad_top=pad_top,
                                                     strideh=strideh,
                                                     stridew=stridew,
                                                     dilationh=dilationh,
                                                     dilationw=dilationw),
            name='fmap_2_col_matrix',
            tag='fmap_2_col_matrix',
            attrs={
                'pad': self.pad,
                'stride': self.stride,
                'dilation': self.dilation,
                'kernel_size': self.weight_shape,
                'load2d_flag': False,
                'group_dict': self.group_dict
            })

    def _fmap_2_matrix_load2d(self, fmap_shape_matrix, fmap):
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
            fmap_width = self.shape_list['fmap_6hd'][4]
            fmap_channel_1 = self.shape_list['fmap_6hd'][2]
            grads_depth = self.shape_list['grads_6hd'][1]
            stride_depth = self.stride[0]
            batch_indices, fmap_c1_indices, hw_mad_indices, fmap_c0_indices = indices
            batch_size_index = batch_indices // grads_depth
            fmap_depth_index = batch_indices % grads_depth * stride_depth + fmap_c1_indices // fmap_channel_1
            fmap_channel_1_index = fmap_c1_indices % fmap_channel_1
            fmap_height_index = (hw_mad_indices // fmap_width)
            fmap_width_index = (hw_mad_indices % fmap_width)
            fmap_c0_index = fmap_c0_indices
            return fmap(batch_size_index, fmap_depth_index,
                        fmap_channel_1_index, fmap_height_index,
                        fmap_width_index, fmap_c0_index)

        return tvm.compute(
            fmap_shape_matrix,
            lambda *indices: __fmap_2_matrix_load2d_compute(indices, fmap),
            name='fmap_2_matrix',
            tag='fmap_2_matrix',
            attrs={
                'pad': self.pad,
                'stride': self.stride,
                'dilation': self.dilation,
                'kernel_size': self.weight_shape,
                "load2d_flag": True,
                'group_dict': self.group_dict
            })

    def _fmap_2_fractal(self, fmap_shape_fmap_matrix, fmap_2_col_matrix,
                       fmap_dtype):
        """
        compute definiton of loading fmap to L0B

        Parameters:
        ----------
        fmap_shape_fmap_matrix : shape of result tensor in L0B
        in shape (batch_size*grads_depth,
                  hw_mad//block_size_K,
                  kernel_depth*fmap_channel_1*kernel_height*kernel_width,
                  fmap_c0,
                  block_size_K)

        fmap_2_col_matrix : input tensor in L1
        in shape (batch_size*grads_depth,
                  grads_height*grads_width,
                  kernel_depth*fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap_dtype : data type of fmap_2_col_matrix


        Returns
        -------
        None
        """
        def __fmap_2_fractal_compute(indices, fmap_2_col_matrix):
            """
            do coordinate calculation

            """

            _, hw_fuse, _, kernel_height, kernel_width, _ = self.shape_list['fmap_original_matrix']

            group_index, n_vm_index, hw_mad_1_indices, fkk_indices, \
                fmap_c0_indices, hw_mad_0_indices = indices

            hw_vm_index = hw_mad_1_indices * _BLOCK_SIZE + hw_mad_0_indices
            c1_vm_index = ((
                (fkk_indices * _BLOCK_SIZE + fmap_c0_indices) // _BLOCK_SIZE) //
                kernel_width) // kernel_height
            depth_index = c1_vm_index // self.group_dict['cin1_g']
            cin_index = c1_vm_index % self.group_dict['cin1_g']
            c1_index = depth_index * self.shape_x_6hd[2] \
                + group_index * self.group_dict['cin1_g'] + cin_index
            kh_vm_index = ((
                (fkk_indices * _BLOCK_SIZE + fmap_c0_indices) // _BLOCK_SIZE) //
                kernel_width) % kernel_height
            kw_vm_index = ((fkk_indices * _BLOCK_SIZE + fmap_c0_indices) //
                           _BLOCK_SIZE) % kernel_width
            c0_vm_index = (fkk_indices * _BLOCK_SIZE + fmap_c0_indices) % _BLOCK_SIZE

            # select padding and 16 align
            return tvm.select(
                tvm.any(hw_vm_index < 0, hw_vm_index > hw_fuse - 1),
                tvm.const(0.0, fmap_dtype),
                fmap_2_col_matrix(n_vm_index, hw_vm_index, c1_index,
                                  kh_vm_index, kw_vm_index, c0_vm_index))

        return tvm.compute(fmap_shape_fmap_matrix,
                           lambda *indices: __fmap_2_fractal_compute(
                               indices, fmap_2_col_matrix),
                           name='fmap_2_col_fractal',
                           tag='fmap_2_col_fractal')

    def _mad(self, mad_shape, grads, fmap):
        """
        calculate mad result tensor
        Parameters
        ----------
        mad_shape : result shape
        (real_g, kernel_depth*fmap_channel_1*kernel_height*kernel_width,
         grads_channel, fmap_c0)

        grads : tensor in L0A
        grads_shape_fractal = (real_g, batch_size*grads_depth,
                               grads_channel_1,
                               hw_mad//block_size_K,
                               grads_c0,
                               block_size_K)

        fmap : tensor in L0B
        fmap_shape_fmap_matrix = (real_g, batch_size*grads_depth,
                                  hw_mad//block_size_K,
                                  kernel_depth*fmap_channel_1* kernel_height*kernel_width,
                                  fmap_c0,
                                  block_size_K)

        Returns
        -------
        None
        """

        batch_size, grads_depth, _, grads_height, grads_width, _ = self.shape_list['grads_6hd']
        hw_fuse = grads_height * grads_width

        batch_axis = tvm.reduce_axis((0, batch_size * grads_depth),
                                     name='axis_b')
        k_axis = tvm.reduce_axis((0, hw_fuse), name='axis_k')

        k_1 = k_axis.var // 16
        k_0 = k_axis.var % 16

        mode = "f162f32"
        if self.res_dtype == "float16":
            mode = "f162f16"

        c_col = tvm.compute(
            mad_shape,
            lambda g, fkk, grads_c, fmap_c0: tvm.sum(
                (grads[g, batch_axis, grads_c // 16, k_1, grads_c % 16, k_0] *
                 fmap[g, batch_axis, k_1, fkk, fmap_c0, k_0]).astype(self.
                                                                  res_dtype),
                axis=[batch_axis, k_axis]),
            name='dw',
            tag="dw",
            attrs={'mode': mode,
                   'kernel_name': self._kernel_name})
        return c_col


@para_check.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (list, tuple),
                             (list, tuple), (list, tuple), dict, (list, tuple),
                             str, str)
def _conv3d_backprop_filter_compute(input_x,
                                   out_backprop,
                                   filter_sizes,
                                   strides=None,
                                   padding=None,
                                   group_dict=None,
                                   dilations=None,
                                   res_dtype="float32",
                                   kernel_name="conv3d_backprop_filter_cce"):
    """
    the DSL interface of conv3d backprop filter compute

    Parameters:
    ----------
    x : the featuremap data, tvm.placeholder, 6hd shape

    out_backprop : the grads data, tvm.placeholder, 6hd shape

    filter_sizes : 5-D shape, specifies the filter sizes

    strides : 3-D shape, specifies in depth, height and width dimension

    padding : 6-D shape, specifies in up/down/left/right dimension

    group_dict ： groups information

    dilations : 5-D shape, specifies in batch/channel/depth/height/width dimension

    res_dtype : the output data type

    Returns
    -------
    result tensor of conv3d_backprop_filter compute

    """
    if not strides:
        strides = [1, 1, 1]
    if not padding:
        padding = [0, 0, 0, 0, 0, 0]
    if not dilations:
        dilations = [1, 1, 1, 1, 1]
    deconv_dw_object = Conv3dBackpropFilter(input_x,
                                            out_backprop,
                                            filter_sizes,
                                            strides=strides,
                                            padding=padding,
                                            group_dict=group_dict,
                                            dilations=dilations,
                                            res_dtype=res_dtype,
                                            kernel_name=kernel_name)
    deconv_dw_object._deconv_dw_access()

    return deconv_dw_object.res_tensor


@para_check.check_input_type(tvm.tensor.Tensor,
                             tvm.tensor.Tensor,
                             (list, tuple),
                             dict)
def conv3d_dw(x,
              out_backprop,
              filter_size,
              para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    x : the featuremap data, tvm.placeholder, 6hd shape

    out_backprop : the grads data, tvm.placeholder, 6hd shape

    filter_size : 5-D shape, specifies the filter sizes

    para_dict : dict of parameters
        strides : 3-D shape, specifies in depth, height and width dimension
        pads : 6-D shape, specifies in up/down/left/right dimension
        dilations : 5-D shape, specifies in batch/channel/depth/height/width dimension
        res_dtype : the output data type
        kernel_name : conv3d_backprop_filter_cce by default
        group_dict : group of parameters

    Returns
    -------
    result tensor of conv3d_backprop_filter compute
    """
    strides = para_dict.get("strides", [1, 1, 1])
    pads = para_dict.get("pads", [0, 0, 0, 0, 0, 0])
    group_dict = para_dict.get("group_dict", None)
    dilations = para_dict.get("dilations", [1, 1, 1, 1, 1])
    res_dtype = para_dict.get("res_dtype", "float32")
    kernel_name = para_dict.get("kernel_name", "conv3d_backprop_filter_cce")

    return _conv3d_backprop_filter_compute(x,
                                          out_backprop,
                                          filter_size,
                                          strides,
                                          pads,
                                          group_dict,
                                          dilations,
                                          res_dtype,
                                          kernel_name)
