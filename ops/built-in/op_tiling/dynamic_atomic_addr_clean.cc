/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <nlohmann/json.hpp>
#include "register/op_tiling.h"
#include "graph/debug/ge_log.h"

namespace optiling {

    const string LOG_INFO = "[INFO] [AtomicAddrClean] ";
    const uint32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
    const uint32_t BYTE_BLOCK = 32;
    const uint32_t BYTE_FP32 = 4;
    const uint32_t MASK_FP32 = 64;
    const uint32_t MAX_REPEAT_TIME = 255;

    struct CleanTilingParams {
        // common input scalar
        int32_t select_key_input_scalar;
        int32_t need_core_num_input_scalar;
        int32_t ele_num_full_mask_full_repeat_time_input_scalar;
        int32_t burst_len_full_mask_full_repeat_time_input_scalar;

        // init input scalar
        // front core
        int32_t ele_num_front_core_input_scalar;
        int32_t init_times_full_mask_full_repeat_time_front_core_input_scalar;
        int32_t ele_num_front_part_front_core_input_scalar;
        int32_t burst_len_last_part_front_core_input_scalar;
        int32_t repeat_time_last_part_front_core_input_scalar;
        // last core
        int32_t ele_num_last_core_input_scalar;
        int32_t init_times_full_mask_full_repeat_time_last_core_input_scalar;
        int32_t ele_num_front_part_last_core_input_scalar;
        int32_t burst_len_last_part_last_core_input_scalar;
        int32_t repeat_time_last_part_last_core_input_scalar;
    };

    int32_t CeilDiv(const int32_t& num,
                    const int32_t& factor)
    {
        int32_t  res;
        res = (num % factor == 0) ? num / factor : num / factor + 1;
        return res;
    }

    bool GetCompileParams(const std::string& op_type,
                          const nlohmann::json &opCompileInfoJson,
                          uint32_t& core_num,
                          uint32_t& ub_size)
    {
        using namespace nlohmann;
        const auto &all_vars = opCompileInfoJson["vars"];
        if (all_vars.count("core_num") == 0) {
            GE_LOGE("op [%s] : get core_num failed", op_type.c_str());
            return false;
        }
        core_num = all_vars["core_num"].get<std::uint32_t>();
        if (all_vars.count("ub_size") == 0) {
            GE_LOGE("op [%s] : get ub_size failed", op_type.c_str());
            return false;
        }
        ub_size = all_vars["ub_size"].get<std::uint32_t>();
        return true;
    }

    void ComputeParamsOneCore(const int32_t& ele_num_one_core,
                              const int32_t& ele_num_full_mask_full_repeat_time_input_scalar,
                              int32_t& init_times_full_mask_full_repeat_time_input_scalar,
                              int32_t& ele_num_front_part_input_scalar,
                              int32_t& burst_len_last_part_input_scalar,
                              int32_t& repeat_time_last_part_input_scalar)
    {
        init_times_full_mask_full_repeat_time_input_scalar = ele_num_one_core / ele_num_full_mask_full_repeat_time_input_scalar;
        ele_num_front_part_input_scalar = init_times_full_mask_full_repeat_time_input_scalar * ele_num_full_mask_full_repeat_time_input_scalar;
        uint32_t ele_num_last_part = ele_num_one_core - ele_num_front_part_input_scalar;
        burst_len_last_part_input_scalar = CeilDiv(ele_num_last_part * BYTE_FP32, BYTE_BLOCK);
        if (ele_num_last_part % MASK_FP32 == 0) {
            repeat_time_last_part_input_scalar = ele_num_last_part / MASK_FP32;
        } else {
            repeat_time_last_part_input_scalar = ele_num_last_part / MASK_FP32 + 1;
        }
    }

    void InitTilingParams(CleanTilingParams& params)
    {
        params.select_key_input_scalar = 0;
        params.need_core_num_input_scalar = 0;
        params.ele_num_full_mask_full_repeat_time_input_scalar = 0;
        params.burst_len_full_mask_full_repeat_time_input_scalar = 0;

        // init input scalar
        // front core
        params.ele_num_front_core_input_scalar = 0;
        params.init_times_full_mask_full_repeat_time_front_core_input_scalar = 0;
        params.ele_num_front_part_front_core_input_scalar = 0;
        params.burst_len_last_part_front_core_input_scalar = 0;
        params.repeat_time_last_part_front_core_input_scalar = 0;
        // last core
        params.ele_num_last_core_input_scalar = 0;
        params.init_times_full_mask_full_repeat_time_last_core_input_scalar = 0;
        params.ele_num_front_part_last_core_input_scalar = 0;
        params.burst_len_last_part_last_core_input_scalar = 0;
        params.repeat_time_last_part_last_core_input_scalar = 0;
    }

    void WriteTilingParams(const CleanTilingParams& params,
                           OpRunInfo& run_info)
    {
        ByteBufferPut(run_info.tiling_data, params.select_key_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.need_core_num_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.ele_num_full_mask_full_repeat_time_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.burst_len_full_mask_full_repeat_time_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.ele_num_front_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.init_times_full_mask_full_repeat_time_front_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.ele_num_front_part_front_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.burst_len_last_part_front_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.repeat_time_last_part_front_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.ele_num_last_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.init_times_full_mask_full_repeat_time_last_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.ele_num_front_part_last_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.burst_len_last_part_last_core_input_scalar);
        ByteBufferPut(run_info.tiling_data, params.repeat_time_last_part_last_core_input_scalar);
    }

    void PrintTilingParams(const std::string& op_type,
                           const CleanTilingParams& params)
    {
        GELOGD("op [%s] : params.select_key_input_scalar=%d", op_type.c_str(), params.select_key_input_scalar);
        GELOGD("op [%s] : params.need_core_num_input_scalar=%d", op_type.c_str(), params.need_core_num_input_scalar);
        GELOGD("op [%s] : params.ele_num_full_mask_full_repeat_time_input_scalar=%d", op_type.c_str(), params.ele_num_full_mask_full_repeat_time_input_scalar);
        GELOGD("op [%s] : params.burst_len_full_mask_full_repeat_time_input_scalar=%d", op_type.c_str(), params.burst_len_full_mask_full_repeat_time_input_scalar);
        GELOGD("op [%s] : params.ele_num_front_core_input_scalar=%d", op_type.c_str(), params.ele_num_front_core_input_scalar);
        GELOGD("op [%s] : params.init_times_full_mask_full_repeat_time_front_core_input_scalar=%d", op_type.c_str(), params.init_times_full_mask_full_repeat_time_front_core_input_scalar);
        GELOGD("op [%s] : params.ele_num_front_part_front_core_input_scalar=%d", op_type.c_str(), params.ele_num_front_part_front_core_input_scalar);
        GELOGD("op [%s] : params.burst_len_last_part_front_core_input_scalar=%d", op_type.c_str(), params.burst_len_last_part_front_core_input_scalar);
        GELOGD("op [%s] : params.repeat_time_last_part_front_core_input_scalar=%d", op_type.c_str(), params.repeat_time_last_part_front_core_input_scalar);
        GELOGD("op [%s] : params.ele_num_last_core_input_scalar=%d", op_type.c_str(), params.ele_num_last_core_input_scalar);
        GELOGD("op [%s] : params.init_times_full_mask_full_repeat_time_last_core_input_scalar=%d", op_type.c_str(), params.init_times_full_mask_full_repeat_time_last_core_input_scalar);
        GELOGD("op [%s] : params.ele_num_front_part_last_core_input_scalar=%d", op_type.c_str(), params.ele_num_front_part_last_core_input_scalar);
        GELOGD("op [%s] : params.burst_len_last_part_last_core_input_scalar=%d", op_type.c_str(), params.burst_len_last_part_last_core_input_scalar);
        GELOGD("op [%s] : params.repeat_time_last_part_last_core_input_scalar=%d", op_type.c_str(), params.repeat_time_last_part_last_core_input_scalar);
    }

    bool CheckSize(const std::string& op_type,
                   const uint32_t& size)
    {
        if (size <= 0) {
            GE_LOGE("op [%s] : workspace size must be greater than 0!", op_type.c_str());
            return false;
        }
        if(size % 32 != 0) {
            GE_LOGE("op [%s] : workspace size must be able to be divided by 32!", op_type.c_str());
            return false;
        }
        return true;
    }

    // tiling function
    bool DynamicAtomicAddrCleanTiling(const std::string& op_type,
                                      const TeOpParas& op_paras,
                                      const nlohmann::json &opCompileInfoJson,
                                      OpRunInfo& run_info)
    {
        GELOGI("op[%s] op tiling begin.", op_type.c_str());
        if (op_paras.const_inputs.find("workspace_size") == op_paras.const_inputs.end()) {
            GE_LOGE("op [%s] : workspace_size not exists", op_type.c_str());
            return false;
        }
        int64_t addr_tensor_size = std::get<1>(op_paras.const_inputs.at("workspace_size"));
        bool flag = CheckSize(op_type, addr_tensor_size);
        if (!flag) {
            return false;
        }
        GELOGD("op [%s] : addr_tensor_size=%d", op_type.c_str(), addr_tensor_size);
        uint32_t ele_num_fp32 = addr_tensor_size / BYTE_FP32;
        CleanTilingParams params;
        // init tiling params
        InitTilingParams(params);
        params.select_key_input_scalar = 1;
        uint32_t core_num = 1;
        uint32_t ub_size = 256 * 1024;
        // get compile_info params
        flag = GetCompileParams(op_type, opCompileInfoJson, core_num, ub_size);
        if (!flag) {
            GE_LOGE("op [%s] : GetCompileParams failed", op_type.c_str());
            return false;
        }
        GELOGI("op[%s] GetCompileParams success.", op_type.c_str());
        // is using all core
        if (addr_tensor_size >= MIN_ELE_SIZE_USING_ALL_CORE) {
            params.need_core_num_input_scalar = core_num;
        } else {
            params.need_core_num_input_scalar = 1;
        }
        // compute tiling params
        params.ele_num_full_mask_full_repeat_time_input_scalar = MASK_FP32 * MAX_REPEAT_TIME;
        params.burst_len_full_mask_full_repeat_time_input_scalar = params.ele_num_full_mask_full_repeat_time_input_scalar * BYTE_FP32 / BYTE_BLOCK;
        if (params.need_core_num_input_scalar == 1) {
            // use one core
            params.ele_num_front_core_input_scalar = ele_num_fp32;
            ComputeParamsOneCore(params.ele_num_front_core_input_scalar,
                                 params.ele_num_full_mask_full_repeat_time_input_scalar,
                                 params.init_times_full_mask_full_repeat_time_front_core_input_scalar,
                                 params.ele_num_front_part_front_core_input_scalar,
                                 params.burst_len_last_part_front_core_input_scalar,
                                 params.repeat_time_last_part_front_core_input_scalar);

            params.ele_num_last_core_input_scalar = params.ele_num_front_core_input_scalar;
            ComputeParamsOneCore(params.ele_num_last_core_input_scalar,
                                 params.ele_num_full_mask_full_repeat_time_input_scalar,
                                 params.init_times_full_mask_full_repeat_time_last_core_input_scalar,
                                 params.ele_num_front_part_last_core_input_scalar,
                                 params.burst_len_last_part_last_core_input_scalar,
                                 params.repeat_time_last_part_last_core_input_scalar);
        } else if (params.need_core_num_input_scalar > 1) {
            // use all core
            // front core
            params.ele_num_front_core_input_scalar = ele_num_fp32 / params.need_core_num_input_scalar;
            ComputeParamsOneCore(params.ele_num_front_core_input_scalar,
                                 params.ele_num_full_mask_full_repeat_time_input_scalar,
                                 params.init_times_full_mask_full_repeat_time_front_core_input_scalar,
                                 params.ele_num_front_part_front_core_input_scalar,
                                 params.burst_len_last_part_front_core_input_scalar,
                                 params.repeat_time_last_part_front_core_input_scalar);
            // last core
            params.ele_num_last_core_input_scalar = ele_num_fp32 - params.ele_num_front_core_input_scalar * (params.need_core_num_input_scalar - 1);
            ComputeParamsOneCore(params.ele_num_last_core_input_scalar,
                                 params.ele_num_full_mask_full_repeat_time_input_scalar,
                                 params.init_times_full_mask_full_repeat_time_last_core_input_scalar,
                                 params.ele_num_front_part_last_core_input_scalar,
                                 params.burst_len_last_part_last_core_input_scalar,
                                 params.repeat_time_last_part_last_core_input_scalar);
        }
        // write tiling params to run_info
        WriteTilingParams(params, run_info);
        // print tiling params
        PrintTilingParams(op_type, params);
        // block_dim, core num used in tik op
        run_info.block_dim = params.need_core_num_input_scalar;
        // workspace, null for tik op
        std::vector<int64_t> workspace;
        run_info.workspaces = workspace;
        GELOGI("op[%s] op tiling success", op_type.c_str());
        return true;
    }
    REGISTER_OP_TILING_FUNC(DynamicAtomicAddrClean, DynamicAtomicAddrCleanTiling);
}
