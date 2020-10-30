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
rl bank, generate schedule from built-in bank or custom bank
"""
import datetime
import json
import os
import traceback

from te import platform as cceconf
from te import tvm
from te.lang.cce.rl_bank.bank_cfg import DTYPE_INDEX
from te.lang.cce.rl_bank.bank_cfg import TAG_INDEX
from te.lang.cce.rl_bank.bank_cfg import SPEC_ATTR_KEY
from te.platform import log
from te.lang.cce.rl_bank.withdraw import gen_sch_by_cheque
from te.lang.cce.rl_bank.withdraw import withdraw
from te.platform.fusion_manager import fusion_manager
from te.lang.cce.te_compute.util import shape_to_list
from te.lang.cce.te_schedule.util import gen_dfs_tensor_map
from te.lang.cce.te_schedule.util import get_reduce_axis_num

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

RL_BANK_DICT = {}
# rl bank version should include te/pass/rl version
RL_BANK_VERSION = 'v001'
DEFAULT_OPP_PATH = '/usr/local/Ascend/opp'


def gen_attr_feature(tensor, curr_tensor_info_list):
    """
    gen_attr_feature
    :param tensor:
    :param curr_tensor_info_list:
    :return:
    """
    if bool(tensor.op.attrs) and any([attr in tensor.op.attrs for attr in SPEC_ATTR_KEY]):
        curr_tensor_info_list.append([])
        for attr in SPEC_ATTR_KEY:
            if attr in tensor.op.attrs:
                curr_tensor_info_list[-1].append(
                    [SPEC_ATTR_KEY.index(attr), tensor.op.attrs[attr]])


def gen_axis_feature(tensor, curr_tensor_info_list):
    """
    gen_axis_feature
    :param tensor:
    :param curr_tensor_info_list:
    :return:
    """
    if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
        # Placeholder has no axis, use shape
        curr_tensor_info_list.append(shape_to_list(tensor.op.shape))
        curr_tensor_info_list.append([])
    else:
        # output shape
        curr_tensor_info_list.append([axis.dom.extent.value for axis in tensor.op.axis])
        # reduce axis idx
        reduce_axis_idx_list = []
        try:
            if tensor.op.reduce_axis:
                reduce_axis_idx_list = get_reduce_axis_num(tensor)
        except Exception:  # pylint: disable=broad-except
            pass
        curr_tensor_info_list.append(reduce_axis_idx_list)


def gen_tensor_feature(dfs_tensor_list):
    """
    gen_tensor_feature
    :param dfs_tensor_list:
    :return:
    """
    feature_list = []
    depends_map = {}
    # get all tensor
    for tensor_idx, tensor in enumerate(dfs_tensor_list):
        if not isinstance(tensor, tvm.tensor.Tensor):
            return ""
        # olny support for PlaceholderOp and ComputeOp
        if not isinstance(tensor.op, (tvm.tensor.PlaceholderOp, tvm.tensor.ComputeOp)):
            return ""
        curr_tensor_info_list = []
        # ===tag===
        op_tag = tensor.op.tag.split("|")[0] if tensor.op.tag else ""
        if op_tag not in TAG_INDEX:
            log.debug("op_tag %s not in TAG_INDEX", op_tag)
            return ""
        curr_tensor_info_list.append(TAG_INDEX[op_tag])

        # op attrs info
        gen_attr_feature(tensor, curr_tensor_info_list)

        # ===axis===
        gen_axis_feature(tensor, curr_tensor_info_list)

        # ===dtype===
        curr_tensor_info_list.append(DTYPE_INDEX[tensor.op.output(0).dtype])

        # ===depens===
        for input_tensor in tensor.op.input_tensors:
            depends_map.setdefault(input_tensor.op.name, []).append(tensor_idx)
        curr_tensor_info_list.append(depends_map.get(tensor.op.name, []))

        feature_list.append(curr_tensor_info_list)

    return str(feature_list)


def get_rl_bank_key(output_tensors_list, op_info=None):
    """
    generate bank_key from tensor info
    tensor_info_list, one for one tensor:tag，output shape，
    output dtype，reduce_axis，depends
    :param output_tensors_list:
    :param op_info:
    :return:bank_key:
    """
    def _get_rl_bank_key(output_tensors, op_info=None):
        # try to get dfs_tensor_list from op_info by auto_schedule
        try:
            dfs_tensor_list = []
            if op_info and op_info.get("dfs_tensor_list", None):
                dfs_tensor_list = op_info["dfs_tensor_list"]
            if not dfs_tensor_list:
                dfs_tensor_list = get_dfs_tensor_list(output_tensors)

            bank_key = gen_tensor_feature(dfs_tensor_list)
            log.debug("bank_key:%s", bank_key)
            return bank_key
        except Exception:  # pylint: disable=broad-except
            log.info("get bank_key fail:%s", traceback.format_exc())
            return ""

    final_bank_key = ""
    if isinstance(output_tensors_list, list) and isinstance(
            output_tensors_list[0], list):
        for output_tensors in output_tensors_list:
            curr_bank_key = _get_rl_bank_key(output_tensors, op_info=op_info)
            if not curr_bank_key:
                return ""
            final_bank_key += curr_bank_key
    else:
        return _get_rl_bank_key(output_tensors_list, op_info=op_info)
    return final_bank_key


def trans_bank_dict(ori_dict, direction):
    """
    trans_from_bank_dict
    :param ori_dict:
    :param direction:
    :return:
    """
    trans_func = json.loads if direction == "from" else json.dumps
    new_dict = {}
    for key, value in ori_dict.items():
        if value == 'tmpl' and direction == "from":
            new_dict[key] = ('tmpl', 0)
        else:
            new_dict[key] = trans_func(value)
    return new_dict


def write_bank(bank_json_path, bank_content):
    """
    write bank
    :param bank_json_path:
    :param base_key_actions_dict:
    :return:
    """
    if not os.path.exists(os.path.dirname(bank_json_path)):
        os.makedirs(os.path.dirname(bank_json_path), 0o750)
    with open(bank_json_path, 'w') as outfile:
        json.dump(bank_content, outfile, sort_keys=True, indent=4)
    os.chmod(bank_json_path, 0o640)


def add_case(outputs, cheque, tick, bank_json_path):
    """
    add cheque to specify bank path
    :param outputs:
    :param cheque:
    :param tick:
    :param bank_json_path:
    :return:
    """
    bank_key = get_rl_bank_key(outputs)
    if not bank_key:
        return False
    if os.path.exists(bank_json_path):
        with open(bank_json_path) as json_file:
            base_key_actions_dict = json.load(json_file)
    else:
        base_key_actions_dict = {}
    # both of key and value are string
    if bank_key in base_key_actions_dict:
        _, old_tick = json.loads(base_key_actions_dict[bank_key])
        if old_tick and old_tick <= tick:
            log.warn(
                "old_tick:%s new_tick:%s, better cheque has been in bank!",
                old_tick, tick)
            return False

    if isinstance(outputs, list) and isinstance(outputs[0], list):
        cheque_list = []
        for i, output_tensors in enumerate(outputs):
            cheque_list += cheque[i]
            curr_bank_key = get_rl_bank_key(output_tensors)
            if not curr_bank_key:
                continue
            base_key_actions_dict.update(
                {curr_bank_key: json.dumps((cheque[i], tick))})
    else:
        cheque_list = cheque
    base_key_actions_dict.update({bank_key: json.dumps((cheque_list, tick))})
    write_bank(bank_json_path, base_key_actions_dict)
    return True


def add_tmpl_to_custom_bank(bank_key, bank_file):
    """
    add tmpl to custom bank. It means use auto_schedule.
    It should not add to builtin bank.
    :param bank_key:
    :param bank_file:
    :return:
    """
    bank_type = 'custom'
    spec_valid, bank_dir = get_custom_rl_path()
    # if assign TUNE_BANK_PATH, custom bank path is TUNE_BANK_PATH/soc_version/rl
    if spec_valid:
        bank_type = "rl"

    soc_version = cceconf.get_soc_spec("SOC_VERSION")
    bank_json_path = os.path.join(bank_dir, soc_version, bank_type, "%s.json" % bank_file)

    if os.path.exists(bank_json_path):
        with open(bank_json_path) as json_file:
            bank_content = json.load(json_file)
    else:
        bank_content = {}
    # both of key and value are string
    if bank_key not in bank_content:
        bank_content[bank_key] = 'tmpl'
    write_bank(bank_json_path, bank_content)
    return True


def get_dfs_tensor_list(out_tensors):
    """
    get_dfs_tensor_list
    :param out_tensors:
    :return:
    """
    if not isinstance(out_tensors, list):
        out_tensors = [out_tensors]

    dfs_tensor_list, _, _, _ = gen_dfs_tensor_map(out_tensors)

    return dfs_tensor_list


def get_custom_rl_path():
    '''
    get_custom_bank_path
    :return:
    '''
    spec_bank = os.getenv("TUNE_BANK_PATH", "")  # pylint: disable=invalid-envvar-default
    base_dir = ''
    spec_valid = False
    if spec_bank:
        spec_bank = os.path.realpath(spec_bank)
        if os.path.isdir(spec_bank) and os.access(spec_bank, os.R_OK | os.W_OK | os.X_OK):
            base_dir = spec_bank
            spec_valid = True

    # if not assign custom bank dir, use default bank dir
    if not base_dir:
        base_dir = get_default_rl_path(custom=True)
        if spec_bank:
            log.warn("TUNE_BANK_PATH:%s is without access,"
                     "use default %s instead!",
                     spec_bank,
                     base_dir)

    return spec_valid, base_dir


def get_default_rl_path(custom=False):
    """
    get_default_bank_path
    :return:
    """
    # parse ASCEND_OPP_PATH to get bank install path
    opp_path = os.getenv("ASCEND_OPP_PATH", "")
    if not opp_path:
        opp_path = DEFAULT_OPP_PATH

    if not os.path.exists(opp_path) or custom:
        base_dir = get_old_rl_path()
    else:
        base_dir = os.path.realpath(os.path.join(opp_path, "data/rl"))

    log.debug("default_rl_path:%s", base_dir)
    return base_dir


def get_old_rl_path():
    """

    :return:
    """
    base_dir = ""
    # parse LD_LIBRARY_PATH to get bank install path
    ld_library_env = os.getenv("LD_LIBRARY_PATH", "")
    for env_item in ld_library_env.split(":"):
        env_item = env_item.strip()
        if not os.path.exists(env_item):
            continue
        if env_item.endswith("/fwkacllib/lib64") \
                or env_item.endswith("/atc/lib64"):
            base_dir = env_item[:-5]
            break
        elif env_item.endswith("/fwkacllib/lib64/") \
                or env_item.endswith("/atc/lib64/"):
            base_dir = env_item[:-6]
            break
    if not base_dir:
        # fwkacllib or atc not in env LD_LIBRARY_PATH
        return ""
    base_dir = os.path.realpath(os.path.join(base_dir, "data/rl"))
    log.debug("old default_rl_path:%s", base_dir)
    return base_dir


def custom_bank_name_match(json_name, bank_name):
    """

    :param json_name:
    :param bank_name:
    :return:
    """
    if not json_name.endswith('.json'):
        return False

    if json_name.startswith(bank_name):
        return True
    # 兼容之前的CustomBank
    if bank_name.startswith("Ascend910A_")and \
        json_name.startswith(bank_name.replace('Ascend910A_', "Ascend910_")):
        return True
    if bank_name.startswith("Ascend910B_")and \
        json_name.startswith(bank_name.replace('Ascend910B_', "Ascend910_")):
        return True
    return False


def read_custom_bank(custom_bank_dir, bank_name):
    """
    read custom bank, maybe there are more than 1 files.
    :param custom_bank_dir:
    :param bank_name:
    :return:
    """
    custom_bank = {}
    bank_files = []
    for bank_json in os.listdir(custom_bank_dir):
        if custom_bank_name_match(bank_json, bank_name):
            bank_file = os.path.join(custom_bank_dir, bank_json)
            bank_files.append(bank_file)
            with open(bank_file) as fh_bank:
                tmp_bank = json.load(fh_bank)
            tmp_bank = trans_bank_dict(tmp_bank, "from")
            for key, (_, tick) in tmp_bank.items():
                if key in custom_bank and \
                        custom_bank[key][1] <= tick and \
                        custom_bank[key][1] > 0:
                    continue
                custom_bank[key] = tmp_bank[key]
    return custom_bank, bank_files


def update_bank_dict(bank_update_to, bank_update_from):
    """

    :param bank_update_to:
    :param bank_update_from:
    :return:
    """
    for key, (_, tick) in bank_update_from.items():
        if key not in bank_update_to \
                or (key in bank_update_to and tick < bank_update_to[key][1]):
            bank_update_to[key] = bank_update_from[key]


def merge_custom_bank(custom_bank_dir, bank_files, bank_name, custom_bank):
    """
    if custom bank files > 1，merge them
    :param custom_bank_dir:
    :param bank_files: origin custom bank files
    :param bank_name: bank_name
    :param custom_bank: custom bank dict
    :return:
    """
    if len(bank_files) > 1:
        for bank_file in bank_files:
            os.remove(bank_file)
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        custom_bank_file = '{}_{}.json'.format(bank_name, time_stamp)
        custom_bank_path = os.path.join(custom_bank_dir, custom_bank_file)
        custom_bank = trans_bank_dict(custom_bank, "to")
        with open(custom_bank_path, 'w') as bank_fh:
            json.dump(custom_bank, bank_fh, sort_keys=True, indent=4)
        os.chmod(custom_bank_path, 0o640)


def get_bank_name():
    """
    get bank name
    :return:
    """
    full_soc_version = cceconf.get_soc_spec("FULL_SOC_VERSION")
    aicore_type = cceconf.get_soc_spec(cceconf.AICORE_TYPE)
    core_num = cceconf.get_soc_spec(cceconf.CORE_NUM)
    bank_name = '{}_{}_{}_{}'.format(full_soc_version, aicore_type,
                                       core_num, RL_BANK_VERSION)
    return bank_name


def satisfy_bank(base_tick, tick, check_type='in'):
    """
    如果想要加入Bank，则需要满足以下条件：
    1，绝对值相差大于5us
    2，绝对值相差大于100或比例小于0.85
    如果想要留在Bank，则只需要满足
    1，绝对值相差大于20或比例小于0.9
    :param base_tick:
    :param tick:
    :param check_type: in or stay。
    :return:
    """
    if check_type == 'in':
        if tick and \
                base_tick and \
                base_tick - tick >= 5 and \
                (base_tick - tick >= 100 or tick / base_tick <= 0.85):
            return True
    else:
        if tick and \
                base_tick and \
                (base_tick - tick >= 20 or tick / base_tick <= 0.9):
            return True
    return False


def get_bank_dict(bank_name, soc_version, force_update=False):
    """
    get_bank_dict
    :return:
    """
    # if already get RL_BANK_DICT, just return
    if bank_name in RL_BANK_DICT and not force_update:
        return True
    RL_BANK_DICT[bank_name] = {}

    # if assign TUNE_BANK_PATH, custom bank path is BANK_PATH/soc_version/rl
    spec_valid, custom_bank_base_dir = get_custom_rl_path()
    custom_bank_dirs = []
    if spec_valid:
        # 指定tune_bank_path的情况，为了兼容老版本，同时读取3个路径：
        # 1,<TUNE_BANK_PATH>/<SOC_VERSION>/rl
        # 2,<TUNE_BANK_PATH>/<SOC_VERSION>/custom
        # 3,/usr/local/Ascend/atc/data/rl/<SOC_VERSION>/custom
        for bank_type in ['custom', 'rl']:
            custom_bank_dirs.append(os.path.join(custom_bank_base_dir,
                                                 soc_version,
                                                 bank_type))

        custom_bank_dirs.append(os.path.join(get_default_rl_path(custom=True),
                                             soc_version,
                                             bank_type))

    else:
        custom_bank_dirs.append(os.path.join(custom_bank_base_dir,
                                             soc_version,
                                             'custom'))

    custom_bank = {}
    for custom_bank_dir in custom_bank_dirs:
        if os.path.isdir(custom_bank_dir):
            cur_custom_bank, bank_files = read_custom_bank(custom_bank_dir,
                                                           bank_name)
            merge_custom_bank(custom_bank_dir,
                              bank_files,
                              bank_name,
                              cur_custom_bank)
            update_bank_dict(custom_bank, cur_custom_bank)

    if custom_bank:
        RL_BANK_DICT[bank_name]["custom"] = custom_bank

    built_in_base_dir = get_default_rl_path()
    built_in_bank_path = os.path.join(built_in_base_dir, soc_version, "built-in/%s.json" % bank_name)

    if not os.path.exists(built_in_bank_path):
        built_in_base_dir = get_default_rl_path(custom=True)
        built_in_bank_path = os.path.join(built_in_base_dir, soc_version, "built-in/%s.json" % bank_name)
    if os.path.exists(built_in_bank_path):
        with open(built_in_bank_path) as json_file:
            RL_BANK_DICT[bank_name]["built-in"] = trans_bank_dict(json.load(json_file), "from")
    # check if bank dict is empty
    return bool(RL_BANK_DICT[bank_name])


def get_cheque(out_tensors, op_info=None, fill_tmpl=False):
    """
    get_cheque_from_bank
    :param out_tensors:op compute output tensors
    :param op_info:op info contain dfs tensor list
    :return:
    """
    if os.getenv("ENABLE_TUNE_BANK", "True").lower() == "false":  # pylint: disable=invalid-envvar-default
        return []

    soc_version = cceconf.get_soc_spec(cceconf.SOC_VERSION)
    bank_name = get_bank_name()
    ret = get_bank_dict(bank_name, soc_version)
    if not ret:
        return []

    # get_rl_bank_key by op_name
    rl_bank_key = get_rl_bank_key(out_tensors, op_info=op_info)
    if not rl_bank_key:
        return []

    # get cheque from BANK_CACHE
    from te.domain.tiling.get_tiling import BANK_CACHE  # pylint: disable=import-outside-toplevel
    if BANK_CACHE:
        try:
            spec_cheque = json.loads(BANK_CACHE)
            if isinstance(spec_cheque, dict) and spec_cheque.get("rl_cheque", {}).get(
                    rl_bank_key, []):
                cheque = spec_cheque["rl_cheque"][rl_bank_key]
                if cheque:
                    log.info("get cheque from BANK_CACHE %s", cheque)
                    return cheque
        except json.decoder.JSONDecodeError:
            log.warn("RL BANK_CACHE:%s must be json format!", BANK_CACHE)

    # get cheque from bank
    cheque, _ = RL_BANK_DICT.get(bank_name, {}).get("custom", {}).get(rl_bank_key, ([], 0))
    if cheque:
        log.info("hit custom bank!")
        return cheque

    cheque, _ = RL_BANK_DICT.get(bank_name, {}).get("built-in", {}).get(rl_bank_key, ([], 0))
    if cheque:
        log.info("hit built-in bank!")

    # 如果Bank不存在这个Key，且fill_tmpl为True，则在Bank中添加一个tmpl的cheque
    if not cheque and fill_tmpl:
        add_tmpl_to_custom_bank(rl_bank_key, bank_name)

    return cheque if cheque else []


def query_rl_bank(out_tensors, op_info=None):
    """
    query_rl_bank
    :param out_tensors:op compute output tensors
    :param op_info:op info contain dfs tensor list
    :return:
    """
    try:
        cheque = get_cheque(out_tensors, op_info=op_info)
        if cheque == 'tmpl':
            return True, None
        elif cheque:
            ret, rl_schedule_obj = gen_sch_by_cheque(out_tensors, cheque)
            if ret:
                return True, rl_schedule_obj
        return False, None
    except:  # pylint: disable=bare-except
        log.warn("get bank_key error:%s", traceback.format_exc())
        return False, None


def update_bank():
    """
    update_bank
    :return:
    """
    soc_version = cceconf.get_soc_spec(cceconf.SOC_VERSION)
    bank_name = get_bank_name()
    ret = get_bank_dict(bank_name, soc_version, force_update=True)
    return ret


def tik_dsl_bank_proc(output_tensors):
    """
    tik dsl op get res and bank
    :param output_tensors:
    :return: 
    """
    if fusion_manager.get_build_cfg() == "disable":
        res_op = []
        if isinstance(output_tensors, list):
            for res in output_tensors:
                res_op.append(res.op)
        else:
            res_op.append(output_tensors.op)
        sch = tvm.create_schedule(res_op)
        return output_tensors, sch

    res_index = fusion_manager.get_res_index()
    cheque_list = fusion_manager.get_cheque_list(res_index=res_index)
    if cheque_list:
        sch, _ = withdraw(output_tensors, cheque_list)
        return output_tensors, sch

    # for RL tune getting op res
    fusion_manager.set_op_res(output_tensors)

    ret, sch = query_rl_bank(output_tensors)
    if ret and sch:
        return output_tensors, sch

    return output_tensors, None
