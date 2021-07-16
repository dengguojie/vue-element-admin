#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
errormgr
"""
from .error_manager_util import get_error_message
from .error_manager_util import raise_runtime_error
from .error_manager_util import raise_runtime_error_cube

from .error_manager_vector import raise_err_input_value_invalid
from .error_manager_vector import raise_err_miss_mandatory_parameter
from .error_manager_vector import raise_err_input_param_not_in_range
from .error_manager_vector import raise_err_input_dtype_not_supported
from .error_manager_vector import raise_err_check_params_rules
from .error_manager_vector import raise_err_input_format_invalid
from .error_manager_vector import raise_err_inputs_shape_not_equal
from .error_manager_vector import raise_err_inputs_dtype_not_equal
from .error_manager_vector import raise_err_input_shape_invalid
from .error_manager_vector import raise_err_two_input_shape_invalid
from .error_manager_vector import raise_err_two_input_dtype_invalid
from .error_manager_vector import raise_err_two_input_format_invalid
from .error_manager_vector import raise_err_specific_reson
from .error_manager_vector import raise_err_pad_mode_invalid
from .error_manager_vector import raise_err_input_param_range_invalid
from .error_manager_vector import raise_err_dtype_invalid

from .error_manager_cube import raise_err_one_para
from .error_manager_cube import raise_err_two_paras
from .error_manager_cube import raise_err_three_paras
from .error_manager_cube import raise_err_four_paras
from .error_manager_cube import raise_err_input_params_not_expected
from .error_manager_cube import raise_err_input_params_not_supported
from .error_manager_cube import raise_err_input_format_invalid
from .error_manager_cube import raise_err_attr_range_invalid
from .error_manager_cube import raise_err_should_4d
from .error_manager_cube import raise_err_specific
from .error_manager_cube import raise_err_common
from .error_manager_cube import raise_err_should_be_4d
from .error_manager_cube import raise_err_specific_user
from .error_manager_cube import raise_err_input_mem_type
from .error_manager_cube import raise_err_output_mem_type
from .error_manager_cube import raise_err_check_the_validity_of_variable
from .error_manager_cube import raise_err_check_the_validity_of_one_variable
from .error_manager_cube import raise_err_specific_input_shape
from .error_manager_cube import raise_err_value_or_format_invalid
from .error_manager_cube import raise_err_equal_invalid
from .error_manager_cube import raise_err_scene_limitation
from .error_manager_cube import raise_err_check_type
from .error_manager_cube import raise_err_scene_equal_limitation
from .error_manager_cube import raise_err_contain_key_invalid
from .error_manager_cube import raise_invalid_range
from .error_manager_cube import raise_err_tiling_type_invalid
from .error_manager_cube import raise_err_info_dict_type_invalid
from .error_manager_cube import raise_err_miss_keyword_invalid
from .error_manager_cube import raise_err_input_invalid
from .error_manager_cube import raise_err_param_type_invalid
from .error_manager_cube import raise_err_param_length_invalid
from .error_manager_cube import raise_err_input_not_support_invalid
from .error_manager_cube import raise_err_input_only_support_invalid
from .error_manager_cube import raise_err_valid_size_invalid
from .error_manager_cube import raise_err_current_value_invalid
from .error_manager_cube import raise_err_previous_value_invalid
from .error_manager_cube import raise_err_previous_current_value_invalid
from .error_manager_cube import raise_err_tuning_tiling_invalid
from .error_manager_cube import raise_err_return_tiling_invalid
from .error_manager_cube import raise_err_dynamic_tiling_mode_invalid
from .error_manager_cube import raise_err_return_value_invalid
