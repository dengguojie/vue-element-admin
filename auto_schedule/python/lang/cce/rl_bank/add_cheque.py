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
add cheque to specify bank
"""
import os
import pickle
import traceback
from te import platform as cceconf
from te.lang.cce.rl_bank.rl_bank import add_case
from te.lang.cce.rl_bank.cheque import gen_cheque
from te.lang.cce.rl_bank.rl_bank import get_default_rl_path
from te.lang.cce.rl_bank.rl_bank import get_custom_rl_path
from te.lang.cce.rl_bank.withdraw import withdraw
from te.platform import log
from te.lang.cce.rl_bank.rl_bank import satisfy_bank


def get_output_tensors(output_tensors, output_names, load_obj):
    """
    get real output tensors
    :param output_tensors:
    :param output_names:
    :param load_obj:
    :return:
    """
    for output_name in output_names:
        for i in range(len(load_obj.stages)):
            stage = load_obj.stages[i]
            # support for tuple_reduce_sum
            if output_name.startswith(stage.op.name + '_v'):
                out_idx = int(output_name.split('_v')[-1])
                output_tensors.append(stage.op.output(out_idx))
            elif output_name == stage.op.name:
                out_idx = 0
                output_tensors.append(stage.op.output(out_idx))


def diff_code(output_tensors, cheque_list, real_schedule_code):
    """
    Helper function. Returns a string containing the unified diff of two list
    :param expected:
    :param actual:
    :param real_schedule_code:
    :return:
    """
    #  gen_sch_by_cheque
    _, draw_code_lines = withdraw(output_tensors, cheque_list)
    print("=" * 80)
    print("\n".join(draw_code_lines))
    print("=" * 80)
    import difflib
    diff = difflib.unified_diff(draw_code_lines, real_schedule_code)
    diff_output = '\n'.join(diff)
    if diff_output:
        print(diff_output)


def get_outputs(code_line_list):
    """
    get_outputs
    :param shcedule_code_list:
    :return:
    """
    output_tensors = []
    output_names = []
    schedule_code = False
    real_schedule_code = []
    for code_line in code_line_list:
        if not code_line.strip():
            continue
        if "#op_outputs:" in code_line:
            output_names = [
                output.strip() for output in code_line.split("#op_outputs:")[1].split(",")
            ]
        elif "pickle.loads(" in code_line:
            tensor_pickle_byte = code_line.split("pickle.loads(b'")[-1][:-2].encode(
                'ISO-8859-1').decode('unicode-escape').encode('ISO-8859-1')
            load_obj = pickle.loads(tensor_pickle_byte)
            get_output_tensors(output_tensors, output_names, load_obj)
        elif "create_schedule" in code_line:
            schedule_code = True
        elif "config = dict()" in code_line or "return sch" in code_line:
            schedule_code = False
        else:
            if schedule_code and not code_line.startswith("    #"):
                real_schedule_code.append(code_line.strip())
    return output_tensors, real_schedule_code


def get_schedule_code(sch_py_path):
    """
    parse schedule py
    :param sch_py_path:
    :return:
    """
    log.info("add cheque sch_py_path:%s", sch_py_path)
    with open(sch_py_path, 'r') as file_handler:
        file_content = file_handler.read()
    code_lines = file_content.split("\n")
    schedule_codes_list = []
    for i, code_line in enumerate(code_lines):
        if code_line.startswith("def dsl_func_"):
            start = i
        elif code_line == "    return sch":
            end = i
            schedule_codes_list.append(code_lines[start: end + 1])
    if not schedule_codes_list:
        schedule_codes_list = [code_lines]
    return schedule_codes_list


def add_cheque_to_bank(sch_py_path, bank_type, bank_file, kernel_name=""):
    """
    add_cheque_to_bank
    :param sch_py_path:
    :param bank_type:
    :param bank_file:
    :param kernel_name:
    :return:
    """
    if not os.path.exists(sch_py_path):
        raise RuntimeError("%s not exists" % sch_py_path)

    if bank_type not in ["custom", "built-in"]:
        raise RuntimeError("bank_type must be custom or built-in,while is %s" %
                           bank_type)
    schedule_codes_list = get_schedule_code(sch_py_path)
    # maybe start with best_
    best_tick = int(os.path.basename(sch_py_path).strip('best_').split('_')[0])
    base_tick = int(os.path.basename(sch_py_path).strip('best_').split('_')[1])
    if not satisfy_bank(base_tick, best_tick, 'in'):
        raise RuntimeError("base_tick:%s best_tick:%s, satisfy_bank check fail!" %
                           (base_tick, best_tick))

    output_tensors_list = []
    cheques_list = []
    for schedule_codes in schedule_codes_list:
        # get output_tensors
        output_tensors, real_schedule_code = get_outputs(schedule_codes)
        if not output_tensors:
            raise RuntimeError("get output_tensors from schedule py file fail!!!")
        # gen cheque
        cheque_list = gen_cheque(schedule_codes, kernel_name=kernel_name)
        cheques_list.append(cheque_list)
        output_tensors_list.append(output_tensors)
        # show code diff if enable DIFF_CODE
        if os.getenv("DIFF_CODE", "False").lower() == "true":
            diff_code(output_tensors, cheque_list, real_schedule_code)

    if bank_type == 'custom':
        spec_valid, bank_dir = get_custom_rl_path()
        # if assign TUNE_BANK_PATH, custom bank path is TUNE_BANK_PATH/soc_version/rl
        if spec_valid:
            bank_type = "rl"
    else:
        bank_dir = get_default_rl_path()

    soc_version = cceconf.get_soc_spec("SOC_VERSION")

    bank_json_path = os.path.join(bank_dir, soc_version, bank_type, "%s.json" %
                                  bank_file)
    ret = add_case(output_tensors_list, cheques_list, best_tick, bank_json_path)
    if ret:
        return True
    return False


def try_add_cheque(sch_py_path, bank_type, bank_file, kernel_name=""):
    """
    try_add_cheque
    :param sch_py_path:
    :param bank_type:
    :param bank_file:
    :param kernel_name:
    :return:
    """
    try:
        ret = add_cheque_to_bank(sch_py_path,
                                 bank_type,
                                 bank_file,
                                 kernel_name=kernel_name)
        return ret, ""
    except Exception:  # pylint: disable=broad-except
        return False, "sch_py_path:%s add cheque to %s bank fail:%s" % (sch_py_path, bank_type,
                                                                        traceback.format_exc())
