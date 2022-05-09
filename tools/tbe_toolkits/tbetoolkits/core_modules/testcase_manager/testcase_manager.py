#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Universal testcase manager for csv support
"""
# Standard Packages
import csv
import random
import logging
from typing import Any, Dict, List, NoReturn, Optional, TextIO

# Third-Party Packages
from .testcase_structure import UniversalTestcaseStructure
from ...utilities import get
from ...utilities import get_dtype_width
from ...utilities import shape_product
from ...utilities import parse_dtype
from ...utilities import get_networkname_by_testcase_name
from ...utilities import set_process_name
from ...utilities import set_thread_name
from ...utilities import get_global_storage
from ...utilities import eliminate_scalar_shapes
from ..infershape.infershape import shape_inference
from ..infershape.infershape import tensorflow_inference


class PLACEHOLDER:
    """
    Simple Placeholder
    """
    pass


class UniversalTestcaseFactory:
    """
    Universal Testcase Manager
    """

    def __init__(self, file: TextIO):
        """
        Store the whole Testcases in the csv file into memory
        """
        set_process_name("TestcaseManager")
        set_thread_name("Initialization")
        # Raw rows
        self.raw_data = []
        # Headers
        self.header = []
        csvreader = csv.reader(file)
        for row in csvreader:
            row = [column.strip() for column in row]
            self.raw_data.append(row)
        # First line should always be the title
        try:
            self.header = self.raw_data[0]
        except IndexError:
            logging.error("Testcase initialization received IndexError, csv file might be empty!")
            raise
        del self.raw_data[0]

        set_thread_name("HeaderCheckTestcaseName")
        # Testcase name generation
        if "testcase_name" not in self.header:
            logging.warning("Testcase name not found!"
                            " It is important to add a testcase_name in order to identify your testcases")
            self.header.append("testcase_name")
            for idx, row in enumerate(self.raw_data):
                row.append("auto_testcase_name_%d" % (idx + 1))

        set_thread_name("HeaderCheckUnidentifiedHeaders")
        self.ignored_headers = []
        for actual_header in self.header:
            if not UniversalTestcaseStructure.is_legit_header(actual_header):
                logging.warning(f"Detected unidentified header: \"{actual_header}\", ignoring.")
                self.ignored_headers.append(actual_header)

        set_thread_name("HeaderCheckDuplicateHeaders")
        # Check duplicates
        header_check_set = set()
        for actual_header in self.header:
            if actual_header in header_check_set and actual_header not in self.ignored_headers:
                logging.error("Detected duplicate header: %s" % actual_header)
                raise RuntimeError("Detected duplicate header: %s" % actual_header)
            header_check_set.add(actual_header)

        set_thread_name("HeaderParseMapping")
        self.real_header_indexes = {}
        # Get idx of required testcase header in real header
        for header_name in UniversalTestcaseStructure.get_all_legit_headers():
            header_idx = self._get_idx_of_header(header_name)
            self.real_header_indexes[header_name] = header_idx

        set_thread_name("TestcasePreParsing")
        raw_testcases = []

        # Process each line
        for idx, line in enumerate(self.raw_data):
            try:
                sub_result = self.__process(line, idx)
            except:
                logging.exception(f"Failed to process row index {idx}")
                raise
            raw_testcases.append(tuple(sub_result.values()))
        logging.debug(f"Processed {len(raw_testcases)} raw testcases")
        set_thread_name("FinalTestcaseStructureFormation")
        self.input_range_inferable = True
        self.output_range_inferable = True
        self.testcases = self._parse(raw_testcases)
        set_process_name()
        set_thread_name()

    def _get_idx_of_header(self, header_name, searched_tag=None) -> Optional[int]:
        """
        Get header position in actual header sequence
        :param header_name: string name of the header
        :param searched_tag: DO NOT USE
        :return: index of the header or its equivalent
        """
        result = None
        if header_name in self.header:
            result = self.header.index(header_name)
        else:
            equivalents = UniversalTestcaseStructure.get_equivalent_headers(header_name)
            # Header not found, search for equivalent
            if not equivalents:
                return result
            # Initialize recursion variable
            if searched_tag is None:
                searched_tag = []
            for equivalent in equivalents:
                if equivalent not in searched_tag:
                    searched_tag.append(header_name)
                    result = self._get_idx_of_header(equivalent,
                                                     searched_tag)
                    if result is not None:
                        break
        return result

    def __process(self, row: list, row_index: int) -> Dict[str, Any]:
        # Initialize full testcase fields
        result = dict(zip(UniversalTestcaseStructure.get_all_legit_headers(),
                          [PLACEHOLDER() for _ in UniversalTestcaseStructure.get_all_legit_headers()]))
        changed, queue = self.__process_over_result(result, row, row_index)
        while changed:
            changed, queue = self.__process_over_result(result, row, row_index, queue)
        if queue:
            changed, queue = self.__process_over_result(result, row, row_index, queue, True)
        while changed:
            changed, queue = self.__process_over_result(result, row, row_index, queue)
        if queue:
            raise RuntimeError(f"Could not determine value of missing fields: {queue}")
        return result

    def __process_over_result(self, result, row, row_index, result_keys=None, apply_default=False):
        changed = False
        placeholder_queue = []
        keys = result_keys if result_keys else result
        for current_header_name in keys:
            column_value = result[current_header_name]
            # Parsed line
            if not isinstance(column_value, PLACEHOLDER):
                continue
            header_raw_idx = self.real_header_indexes[current_header_name]
            # Header not exist or value is empty, check for equivalent and default value
            if header_raw_idx is None or header_raw_idx >= len(row) or not row[header_raw_idx]:
                if UniversalTestcaseStructure.has_equivalent_header(current_header_name) and not apply_default:
                    # Check for equivalent
                    resolved = False
                    value = None
                    for equivalent_header in UniversalTestcaseStructure.get_equivalent_headers(current_header_name):
                        equivalent_header_value = result[equivalent_header]
                        if isinstance(equivalent_header_value, PLACEHOLDER):
                            # Equivalent not exist either, check for next equivalent
                            continue
                        else:
                            value = equivalent_header_value
                            resolved = True
                            break
                    if not resolved:
                        placeholder_queue.append(current_header_name)
                        continue
                elif UniversalTestcaseStructure.has_default_value(current_header_name):
                    # Check for default value
                    changed = True
                    default_value = str(UniversalTestcaseStructure.get_default_value(current_header_name))
                    result[current_header_name] = \
                        UniversalTestcaseStructure.get_header_func(current_header_name)(default_value)
                    continue
                else:
                    placeholder_queue.append(current_header_name)
                    continue
            else:
                value = row[header_raw_idx]
            try:
                changed = True
                if isinstance(value, str):
                    result[current_header_name] = UniversalTestcaseStructure.get_header_func(current_header_name)(value)
                else:
                    result[current_header_name] = value
            except:
                logging.exception(f"Failed to process header {current_header_name} of row {row_index} with "
                                  f"value {value}")
                raise
        return changed, placeholder_queue

    def get(self) -> Dict[int, set]:
        """
        Get all testcases
        :return:
        """
        return self.testcases

    # noinspection PyBroadException
    def _parse(self, testcases: list) -> Dict[int, set]:
        parsed_testcases: List[UniversalTestcaseStructure] = []
        testcase_names: str = set()
        # For testcase_count selector
        if 0 < get_global_storage().selected_testcase_count < len(testcases):
            logging.info("Selecting %d cases from all testcases" % get_global_storage().selected_testcase_count)
            all_indexes = random.sample(tuple(range(len(testcases))), k=get_global_storage().selected_testcase_count)
            testcases = [testcase for testcase_idx, testcase in enumerate(testcases) if testcase_idx in all_indexes]
        # Iterate through testcases, eliminate duplicates and do shape-range inference
        for testcase_idx, testcase in enumerate(testcases):
            # Construct Testcase Structure
            testcase_struct = UniversalTestcaseStructure()
            testcase_struct.original_line = self.raw_data[testcase_idx]
            testcase_struct.original_header = self.header
            headers = UniversalTestcaseStructure.get_all_legit_headers()
            unidentified_headers = []
            for header_idx, header in enumerate(headers):
                if hasattr(testcase_struct, header):
                    setattr(testcase_struct, header, testcase[header_idx])
                else:
                    unidentified_headers.append(header)
            if unidentified_headers:
                raise KeyError(f"TestcaseManager header not match with UniversalTestcaseStructure, "
                               f"Report Bug to us: {unidentified_headers}")
            # Skip testcase if it is disabled
            if not testcase_struct.is_enabled:
                logging.debug("Testcase %s skipped" % testcase_struct.testcase_name)
                continue
            # Skip testcase if it is not in global testcase_name selector range
            if not self._check_testcase_name_selection(testcase_struct):
                continue
            # Skip testcase if it is not in global testcase_index selector range
            if not self._check_testcase_indexes_selection(testcase_idx, testcase_struct):
                continue
            # Skip testcase if it is not in global testcase_op_name selector range
            if not self._check_testcase_operator_selection(testcase_struct.op_name):
                continue
            ########
            # IMPORTANT !!! enabled testcases will now be recorded !!! IMPORTANT
            ########
            if testcase_struct.testcase_name in testcase_names:
                logging.warning("Detected duplicate testcase name: %s" % testcase_struct.testcase_name)
            testcase_names.add(testcase_struct.testcase_name)
            parsed_testcases.append(testcase_struct)
            set_thread_name(testcase_struct.testcase_name)
            try:
                if not self._check_other_params(testcase_struct):
                    continue
                if not self._parse_stc_input_dtypes(testcase_struct):
                    continue
                if not self._parse_stc_outputs(testcase_struct):
                    continue
                if not self._parse_stc_ori_outputs(testcase_struct):
                    continue
                if not self._parse_output_dtypes(testcase_struct):
                    continue
                self._check_dyn_inputs(testcase_struct)
                self._parse_dyn_input_dtypes(testcase_struct)
                self._check_const_inputs(testcase_struct)
                self._parse_dyn_outputs(testcase_struct)
                self._parse_dyn_ori_outputs(testcase_struct)
                self._parse_dyn_input_ranges(testcase_struct)
                self._parse_dyn_output_ranges(testcase_struct)
                if not self._stc_shape_size_check(testcase_struct):
                    continue
                if not self.__perform_shape_check(testcase_struct):
                    continue
                testcase_struct.network_name = get_networkname_by_testcase_name(testcase_struct.testcase_name)
            except:
                raise RuntimeError(f"Failed parsing testcase {testcase_struct.testcase_name}")
        # Construct final result
        testcase_group_dict = {}
        for parsed_testcase in parsed_testcases:
            hash_index = parsed_testcase.get_compilation_hash()
            testcase_group = testcase_group_dict.setdefault(hash_index, set())
            if parsed_testcase not in testcase_group:
                testcase_group.add(parsed_testcase)
            else:
                logging.warning("Duplicate testcase: %s" % parsed_testcase.testcase_name)
        return testcase_group_dict

    @staticmethod
    def _check_dyn_inputs(testcase_struct) -> NoReturn:
        if testcase_struct.dyn_inputs is None:
            testcase_struct.dyn_is_valid = False
            testcase_struct.dyn_fail_reason = "DYN_INPUT_MISSING"
            testcase_struct.dyn_inputs = testcase_struct.stc_inputs

    @staticmethod
    def _parse_dyn_outputs(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if isinstance(testcase_struct.dyn_outputs, str):
                    testcase_struct.dyn_outputs = do_shape_inference(testcase_struct.dyn_inputs,
                                                                     testcase_struct.dyn_outputs,
                                                                     testcase_struct.other_compilation_params)
                elif testcase_struct.dyn_outputs is None:
                    testcase_struct.dyn_outputs = tensorflow_inference(testcase_struct)
                return
            except:
                logging.warning("Dynamic output shape inference failure, use static output instead.")
        testcase_struct.dyn_outputs = testcase_struct.stc_outputs

    @staticmethod
    def _parse_dyn_ori_outputs(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if isinstance(testcase_struct.dyn_ori_outputs, str):
                    testcase_struct.dyn_ori_outputs = do_shape_inference(testcase_struct.dyn_ori_inputs,
                                                                         testcase_struct.dyn_ori_outputs,
                                                                         testcase_struct.other_compilation_params)
                elif testcase_struct.dyn_ori_outputs is None:
                    testcase_struct.dyn_ori_outputs = testcase_struct.dyn_outputs
                return
            except:
                logging.exception("Dynamic original output shape inference failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_ORI_OUTPUT_INFERENCE_FAILED"
        testcase_struct.dyn_ori_outputs = testcase_struct.stc_ori_outputs

    def _parse_dyn_output_ranges(self, testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if len(testcase_struct.dyn_output_ranges) == 0 or testcase_struct.dyn_output_ranges is None:
                    self.output_range_inferable = self.output_range_inferable and True
                    testcase_struct.dyn_output_ranges = shape_inference(testcase_struct.dyn_outputs, (), "RANGE")
                elif shape_inference(testcase_struct.dyn_outputs, (), "RANGE") == testcase_struct.dyn_output_ranges:
                    self.output_range_inferable = self.output_range_inferable and True
                else:
                    self.output_range_inferable = self.output_range_inferable and False
            except:
                logging.exception("Dynamic output range inference failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_OUTPUT_RANGE_INFERENCE_FAILED"

    @staticmethod
    def _parse_dyn_input_dtypes(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                testcase_struct.dyn_input_dtypes = tuple(parse_dtype(dtype)
                                                         for dtype in testcase_struct.dyn_input_dtypes)
                return
            except:
                logging.exception("Dynamic input dtypes parsing failure, turning dynamic profiling off")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_INPUT_DTYPES_INVALID"
        testcase_struct.dyn_input_dtypes = testcase_struct.stc_input_dtypes

    def _parse_dyn_input_ranges(self, testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if testcase_struct.dyn_input_ranges is None or len(testcase_struct.dyn_input_ranges) == 0:
                    testcase_struct.dyn_input_ranges = shape_inference(testcase_struct.dyn_inputs, (), "RANGE")
                    self.input_range_inferable = self.input_range_inferable and True
                elif shape_inference(testcase_struct.dyn_inputs, (), "RANGE") == testcase_struct.dyn_input_ranges:
                    self.input_range_inferable = self.input_range_inferable and True
                else:
                    self.input_range_inferable = self.input_range_inferable and False
            except:
                logging.exception("Dynamic input range inference failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_INPUT_RANGE_INFERENCE_FAILED"

    @staticmethod
    def _parse_stc_outputs(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            if isinstance(testcase_struct.stc_outputs, str):
                testcase_struct.stc_outputs = do_shape_inference(testcase_struct.stc_inputs,
                                                                 testcase_struct.stc_outputs,
                                                                 testcase_struct.other_runtime_params)
            elif testcase_struct.stc_outputs is None:
                testcase_struct.stc_outputs = tensorflow_inference(testcase_struct)
        except:
            logging.exception("Static output shape inference failure")
            testcase_struct.is_valid = False
            testcase_struct.fail_reason = "STC_OUTPUT_INFERENCE_FAILED"
        return testcase_struct.is_valid

    @staticmethod
    def _parse_stc_ori_outputs(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            if isinstance(testcase_struct.stc_ori_outputs, str):
                testcase_struct.stc_ori_outputs = do_shape_inference(testcase_struct.stc_ori_inputs,
                                                                     testcase_struct.stc_ori_outputs,
                                                                     testcase_struct.other_runtime_params)
            elif testcase_struct.stc_ori_outputs is None:
                testcase_struct.stc_ori_outputs = tensorflow_inference(testcase_struct, True)
        except:
            logging.exception("Static original output shape inference failure")
            testcase_struct.is_valid = False
            testcase_struct.fail_reason = "STC_ORI_OUTPUT_INFERENCE_FAILED"
        return testcase_struct.is_valid

    @staticmethod
    def _check_testcase_name_selection(testcase_struct: UniversalTestcaseStructure) -> bool:
        if get_global_storage().selected_testcases:
            if testcase_struct.testcase_name \
                    in [s for s in get_global_storage().selected_testcases]:
                logging.info("Found selected testcase: %s" % testcase_struct.testcase_name)
                return True
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_indexes_selection(testcase_idx, testcase_struct: UniversalTestcaseStructure) -> bool:
        if get_global_storage().selected_testcase_indexes:
            if testcase_idx in get_global_storage().selected_testcase_indexes:
                logging.info("Found index %d of selected testcase: %s" % (testcase_idx,
                                                                          testcase_struct.testcase_name))
                return True
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_operator_selection(testcase_op_name) -> bool:
        if get_global_storage().selected_operators:
            if testcase_op_name in get_global_storage().selected_operators:
                return True
        else:
            return True
        return False

    @staticmethod
    def __perform_shape_check(testcase_struct: UniversalTestcaseStructure) -> bool:
        if testcase_struct.shape_check and testcase_struct.dyn_is_valid:
            # Static inputs num should never be greater than dynamic inputs num
            if len(testcase_struct.stc_inputs) > len(testcase_struct.dyn_inputs):
                logging.error(
                    "Testcase %s has been disabled because of dynamic-static tensor num difference"
                    % testcase_struct.testcase_name)
                testcase_struct.is_valid = False
                testcase_struct.fail_reason = "STC_TENSOR_GT_DYN_TENSOR"
            # Pre-DYN-STC-Shape check
            if (len(testcase_struct.const_input_indexes) + len(testcase_struct.dyn_inputs) ==
                    len(testcase_struct.stc_inputs)):
                # DYN-STC-Shape check
                const_input_offset = 0
                normalized_stc_shapes = eliminate_scalar_shapes(testcase_struct.stc_inputs)
                for shape_idx, dyn_shape in enumerate(eliminate_scalar_shapes(testcase_struct.dyn_inputs)):
                    if shape_idx in testcase_struct.const_input_indexes:
                        const_input_offset += 1
                    stc_shape = normalized_stc_shapes[shape_idx - const_input_offset]
                    # NoneType stc_shape and -2 dyn_shape should be ignored
                    if stc_shape is None or -2 in dyn_shape:
                        continue
                    if len(dyn_shape) != len(stc_shape):
                        logging.error(
                            "Testcase %s has been disabled because of dynamic-static shape length difference"
                            % testcase_struct.testcase_name)
                        testcase_struct.is_valid = False
                        testcase_struct.fail_reason = "DYN_STC_SHAPE_LEN"
                        break
                    if False in [dyn_dim == stc_dim or dyn_dim == -1
                                 for dyn_dim, stc_dim in zip(dyn_shape, stc_shape)]:
                        logging.error(
                            f"Testcase {testcase_struct.testcase_name} has been disabled because of "
                            f"dynamic-static shape dim difference: {dyn_shape} vs {stc_shape}")
                        testcase_struct.is_valid = False
                        testcase_struct.fail_reason = "DYN_STC_SHAPE_DIFF"
                        break
        return testcase_struct.is_valid

    @staticmethod
    def _stc_shape_size_check(testcase_struct: UniversalTestcaseStructure) -> bool:
        size_limitation = shape_size_check(testcase_struct.stc_inputs, testcase_struct.stc_outputs,
                                           testcase_struct.stc_input_dtypes, testcase_struct.output_dtypes)
        if size_limitation > 0:
            logging.error(f"Testcase {testcase_struct.testcase_name} is invalid because shape is out of bound: "
                          f"{size_limitation}")
            testcase_struct.is_valid = False
            testcase_struct.fail_reason = "STC_SHAPE_OUT_OF_BOUND"
        return testcase_struct.is_valid

    @staticmethod
    def _check_const_inputs(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if len(testcase_struct.const_input_modes) == 0 and len(testcase_struct.const_input_indexes) > 0:
                    testcase_struct.const_input_modes = (None,)
            except:
                logging.exception("Const input modes or indexes parsing failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "CONST_INPUT_INVALID"

    @staticmethod
    def _check_other_params(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            for param in testcase_struct.other_compilation_params:
                if param not in testcase_struct.other_runtime_params:
                    testcase_struct.other_runtime_params[param] = testcase_struct.other_compilation_params[param]
                elif param in testcase_struct.other_runtime_params and \
                        testcase_struct.other_runtime_params[param] != testcase_struct.other_compilation_params[param]:
                    logging.warning(f"Compilation param {param} conflicts with which in runtime params: "
                                    f"{testcase_struct.other_runtime_params[param]} vs "
                                    f"{testcase_struct.other_compilation_params[param]}")
        except:
            logging.exception("Compilation or Runtime params parsing failure")
            testcase_struct.dyn_is_valid = False
            testcase_struct.is_valid = False
            testcase_struct.dyn_fail_reason = testcase_struct.fail_reason = "OTHER_PARAMS_INVALID"
        return testcase_struct.is_valid

    @staticmethod
    def _parse_stc_input_dtypes(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            if testcase_struct.is_valid:
                testcase_struct.stc_input_dtypes = tuple(parse_dtype(dtype)
                                                         for dtype in testcase_struct.stc_input_dtypes)
        except:
            logging.exception("Static input dtypes parsing failure")
            testcase_struct.is_valid = False
            testcase_struct.dyn_is_valid = False
            testcase_struct.fail_reason = "STC_INPUT_DTYPES_INVALID"
        return testcase_struct.is_valid

    @staticmethod
    def _parse_output_dtypes(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            testcase_struct.output_dtypes = tuple(parse_dtype(dtype) for dtype in testcase_struct.output_dtypes)
        except:
            logging.exception("Output dtypes parsing failure")
            testcase_struct.dyn_is_valid = False
            testcase_struct.is_valid = False
            testcase_struct.dyn_fail_reason = testcase_struct.fail_reason = "OUTPUT_DTYPES_INVALID"
        return testcase_struct.is_valid


def do_shape_inference(inputs: tuple, outputs: str, args: dict) -> tuple:
    """
    Shape inference
    :param inputs:
    :param outputs:
    :param args:
    :return:
    """
    if None in inputs:
        raise ValueError("Automatic inference doesn't support None input")
    if outputs in ("ELEWISE",):
        return shape_inference(inputs, (1, None), outputs)
    if outputs in ("REDUCE",):
        if "axes" in args:
            axes = args["axes"]
        elif "axis" in args:
            axes = args["axis"]
        else:
            axes = None
        return shape_inference(inputs, (axes, 1, None), outputs)
    elif "ELEWISE" in outputs:
        try:
            args = eval(outputs[7:])
        except:
            raise ValueError("Unable to parse shape inference args from %s" % outputs)
        else:
            return shape_inference(inputs, args, "ELEWISE")
    elif "REDUCE" in outputs:
        try:
            _args = eval(outputs[6:])
            if "axes" in args:
                args = (args["axes"], *_args[1:])
            elif "axis" in args:
                args = (args["axis"], *_args[1:])
            else:
                args = _args
        except:
            raise ValueError("Unable to parse shape inference args from %s" % outputs)
        else:
            return shape_inference(inputs, args, "REDUCE")
    else:
        raise ValueError("Invalid shape inference value %s" % outputs)


def shape_size_check(shapes: tuple, shapes_out: tuple, dtypes: tuple, dtypes_out: tuple) -> int:
    """
    Check shape product size, return None if shape size out of bound
    :param dtypes_out:
    :param dtypes:
    :param shapes_out:
    :param shapes:
    :return:
    """
    shapes = eliminate_scalar_shapes(shapes)
    shapes_out = eliminate_scalar_shapes(shapes_out)
    total_shape_product_value = 0
    for idx, shape in enumerate(shapes):
        if shape is not None:
            shape_size = shape_product(shape)
            total_shape_product_value += shape_size * get_dtype_width(get(dtypes, idx))

    for idx, shape in enumerate(shapes_out):
        if shape is not None:
            shape_size = shape_product(shape)
            total_shape_product_value += shape_size * get_dtype_width(get(dtypes_out, idx))

    if total_shape_product_value > get_global_storage().DAVINCI_HBM_SIZE_LIMIT:
        return get_global_storage().DAVINCI_HBM_SIZE_LIMIT
    return 0
