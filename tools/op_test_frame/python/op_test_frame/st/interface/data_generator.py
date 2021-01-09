# !/usr/bin/env python
# coding=utf-8
"""
Function:
DataGenerator class
This class mainly involves generate data.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import sys
    import os
    import numpy as np
    import importlib
    from . import utils
    from . import st_report
    from . import op_st_case_info
    from op_test_frame.common import op_status
except ImportError as import_error:
    sys.exit(
        "[data_generator] Unable to import module: %s." % str(import_error))


class DataGenerator:
    """
    The class for data generator.
    """

    def __init__(self, case_list, output_path, cmd_mi, report):
        self.case_list = case_list
        self.report = report
        if cmd_mi:
            self.output_path = os.path.join(output_path, 'run', 'out',
                                            'test_data', 'data')
        else:
            op_name_path = os.path.join(output_path, case_list[0]['op'])
            self.output_path = os.path.join(output_path, op_name_path, 'run',
                                            'out', 'test_data', 'data')

    @staticmethod
    def gen_data(data_shape, min_value, max_value, dtype,
                 distribution='uniform'):
        """
        generate data
        :param data_shape: the data shape
        :param min_value: min value
        :param max_value: max value
        :param dtype: the data type
        :param distribution: the data distribution
        :return: the numpy data
        """
        real_dtype = dtype
        if dtype == np.bool:
            min_value = 0
            max_value = 2  # [0, 2) in uniform
            dtype = np.int8
        if distribution == 'uniform':
            # Returns the uniform distribution random value.
            # min indicates the random minimum value,
            # and max indicates the random maximum value.
            data = np.random.uniform(low=min_value, high=max_value,
                                     size=data_shape).astype(dtype)
        elif distribution == 'normal':
            # Returns the normal (Gaussian) distribution random value.
            # min is the central value of the normal distribution,
            # and max is the standard deviation of the normal distribution.
            # The value must be greater than 0.
            data = np.random.normal(loc=min_value,
                                    scale=abs(max_value) + 1e-4,
                                    size=data_shape).astype(dtype)
        elif distribution == 'beta':
            # Returns the beta distribution random value.
            # min is alpha and max is beta.
            # The values of both min and max must be greater than 0.
            data = np.random.beta(a=abs(min_value) + 1e-4,
                                  b=abs(max_value) + 1e-4,
                                  size=data_shape).astype(dtype)
        elif distribution == 'laplace':
            # Returns the Laplacian distribution random value.
            # min is the central value of the Laplacian distribution,
            # and max is the exponential attenuation of the Laplacian
            # distribution.  The value must be greater than 0.
            data = np.random.laplace(loc=min_value,
                                     scale=abs(max_value) + 1e-4,
                                     size=data_shape).astype(dtype)
        elif distribution == 'triangular':
            # Return the triangle distribution random value.
            # min is the minimum value of the triangle distribution,
            # mode is the peak value of the triangle distribution,
            # and max is the maximum value of the triangle distribution.
            mode = np.random.uniform(low=min_value, high=max_value)
            data = np.random.triangular(left=min_value, mode=mode,
                                        right=max_value,
                                        size=data_shape).astype(dtype)
        elif distribution == 'relu':
            # Returns the random value after the uniform distribution
            # and relu activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            data = np.maximum(0, data_pool)
        elif distribution == 'sigmoid':
            # Returns the random value after the uniform distribution
            # and sigmoid activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            data = 1 / (1 + np.exp(-data_pool))
        elif distribution == 'softmax':
            # Returns the random value after the uniform distribution
            # and softmax activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            data = np.exp(data_pool) / np.sum(np.exp(data_pool))
        elif distribution == 'tanh':
            # Returns the random value after the uniform distribution
            # and tanh activation.
            data_pool = np.random.uniform(low=min_value, high=max_value,
                                          size=data_shape).astype(dtype)
            data = (np.exp(data_pool) - np.exp(-data_pool)) / \
                   (np.exp(data_pool) + np.exp(-data_pool))
        else:
            utils.print_error_log('The distribution(%s) is invalid.')
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_WRITE_FILE_ERROR)
        if real_dtype == np.bool:
            data = data.astype(real_dtype)
        return data

    def generate(self):
        """
        generate data by case list
        """
        utils.check_path_valid(self.output_path, True)
        for case in self.case_list:
            case_name = case['case_name']
            calc_func_params_tmp = list()
            utils.print_info_log(
                'Start to generate the data for %s.' % case_name)
            for index, input_desc in enumerate(case['input_desc']):
                range_min, range_max = input_desc['value_range']
                if input_desc.get('type') in utils.OPTIONAL_TYPE_LIST:
                    continue
                if case.get('st_mode') == "ms_python_train":
                    dtype = utils.DTYPE_TO_MINDSPORE_MAP[input_desc.get('type')]
                else:
                    dtype = utils.DTYPE_TO_NUMPY_MAP[input_desc.get('type')]
                file_path = os.path.join(
                    self.output_path,
                    case_name + '_input_' + str(index) + '.bin')
                if os.path.exists(file_path):
                    utils.print_error_log(
                        'The file %s already exists, please delete it then'
                        ' retry.' % file_path)
                    raise utils.OpTestGenException(
                        utils.OP_TEST_GEN_WRITE_FILE_ERROR)
                try:
                    data = self.gen_data(
                            input_desc['shape'], range_min, range_max, dtype,
                            input_desc['data_distribute'])
                except MemoryError as error:
                    utils.print_warn_log(
                        'Failed to generate data for %s. The shape is too '
                        'large to invoke MemoryError. %s' % (file_path, error))
                    raise utils.OpTestGenException(
                        utils.OP_TEST_GEN_WRITE_FILE_ERROR)
                try:
                    calc_func_params_tmp.append({
                        'value': data})
                    data.tofile(file_path)
                    os.chmod(file_path, utils.WRITE_MODES)
                except OSError as error:
                    utils.print_warn_log(
                        'Failed to generate data for %s. %s' % (
                            file_path, error))
                    raise utils.OpTestGenException(
                        utils.OP_TEST_GEN_WRITE_FILE_ERROR)
            # get attr param
            if case.get('attr'):
                for index, attr in enumerate(case['attr']):
                    calc_func_params_tmp.append({'value': attr.get('value')})
            expect_data_paths = self._generate_expect_data(
                case, calc_func_params_tmp)
            # deal with report
            case_report = self.report.get_case_report(case_name)
            case_report.trace_detail.st_case_info.input_data_paths = \
                self.output_path
            if expect_data_paths:
                case_report.trace_detail.st_case_info.expect_data_paths = \
                    expect_data_paths
                utils.print_info_log(
                    'Finish to generator the expect output data for '
                    '%s.' % case_name)

    def _generate_expect_data(self, case, calc_func_params_tmp):
        expect_data_paths = []
        case_name = case.get('case_name')
        expect_data_dir = os.path.join(self.output_path, 'expect')
        utils.make_dirs(expect_data_dir)
        utils.print_info_log(
            'Start to generate the expect output data for %s.' %
            case_name)
        if case.get("calc_expect_func_file") \
                and case.get("calc_expect_func_file_func"):
            expect_func_file = case["calc_expect_func_file"]
            expect_func = case.get("calc_expect_func_file_func")
            sys.path.append(os.path.dirname(expect_func_file))
            py_file = os.path.basename(expect_func_file)
            module_name, _ = os.path.splitext(py_file)
            utils.print_info_log("Start to import %s in %s." % (module_name,
                                                                py_file))
            module = importlib.import_module(module_name)
            try:
                func = getattr(module, expect_func)
                expect_result_tensors = func(*calc_func_params_tmp)
            except Exception as ex:
                utils.print_error_log(
                    'Failed to execute function "%s" in %s. %s' % (
                        expect_func, expect_func_file, str(ex)))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_PARAM_ERROR)
            if not isinstance(expect_result_tensors, (list, tuple)):
                expect_result_tensors = [expect_result_tensors, ]
            for idx, expect_result_tensor in enumerate(expect_result_tensors):
                output_dtype = case['output_desc'][idx]['type']
                if str(expect_result_tensor.dtype) != output_dtype:
                    utils.print_warn_log("The dtype of expect date clc by "
                                         "%s is %s , is not same as "
                                         "the dtype(%s) in output_desc index("
                                         "%s). "
                                         % (expect_func,
                                            expect_result_tensor.dtype,
                                            output_dtype, str(idx)))
                expect_data_name = "%s_expect_output_%s_%s.bin" % (
                    case_name, str(idx), output_dtype)
                expect_data_path = os.path.join(expect_data_dir,
                                                expect_data_name)
                expect_result_tensor.tofile(expect_data_path)
                utils.print_info_log("Successfully generated expect "
                                     "data:%s." % expect_data_path)
                expect_data_paths.append(expect_data_path)
        return expect_data_paths



