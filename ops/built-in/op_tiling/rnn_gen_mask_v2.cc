#include <string>
#include <math.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling{
  struct RnnGenMaskV2TilingParams
  {
    int32_t cal_mode;
    int32_t core_used;
    int32_t batch_size;
    int32_t num_step;
    int32_t hidden_size;
    int32_t hidden_size_block;
    int32_t repeat;
    int32_t rounds;
    int32_t batch_num_per_aicore;
    int32_t batch_tail;
  };

  void InitTilingParams(RnnGenMaskV2TilingParams &params)
  {
    OP_LOGD("InitTilingParams is running");
    params.cal_mode = 0;
    params.core_used = 0;
    params.batch_size = 0;
    params.num_step = 0;
    params.hidden_size = 0;
    params.hidden_size_block = 0;
    params.repeat = 0;
    params.rounds = 0;
    params.batch_num_per_aicore = 0;
    params.batch_tail = 0;
  }

  bool GetCompileInfoV2(const std::string &op_type, const nlohmann::json &op_compile_info,
                        int32_t &aicore_num, int32_t &block)
  {
    OP_LOGD("GetCompileInfoV2 is running");
    using namespace nlohmann;
    auto all_vars = op_compile_info["vars"];
    if (all_vars.count("available_aicore_num") == 0)
    {
      VECTOR_INNER_ERR_REPORT_TILIING("RnnGenMaskV2Tiling", "GetCompileInfoV2, get available_aicore_num  error");
      return false;
    }
    aicore_num = all_vars["available_aicore_num"].get<std::int32_t>();
    if (all_vars.count("block") == 0)
    {
      VECTOR_INNER_ERR_REPORT_TILIING("RnnGenMaskV2Tiling", "GetCompileInfoV2, get block error");
      return false;
    }
    block = all_vars["block"].get<std::int32_t>();
    return true;
  }

  int32_t CalTilingModeV2(std::vector<int64_t> x_shape)
  {
    OP_LOGD("CalTilingModeV2 is running");
    int32_t tiling_mode = 0;
    auto batch_size = x_shape[1];
    if (batch_size >= 1)
    {
      tiling_mode = 1;
    }
    return tiling_mode;
  }
    
  static void CalCoreInfo(RnnGenMaskV2TilingParams &tiling_params, int32_t & core_num, std::vector<int64_t> & x_shape)
  {
    OP_LOGD("CalCoreInfo is running");
    int32_t batch = x_shape[1];
    int32_t num_step = x_shape[0];
    auto rounds = batch * num_step;
    int32_t batch_num_per_aicore = 0;
    int32_t core_used = 0;
    int32_t batch_tail = 0;

    if (rounds > core_num)
    {
      core_used = core_num;
    }
    else
    {
      core_used = rounds;
    }
    batch_num_per_aicore = rounds / core_used;
    batch_tail = rounds % core_used;

    tiling_params.batch_num_per_aicore = batch_num_per_aicore;
    tiling_params.core_used = core_used;
    tiling_params.batch_tail = batch_tail;
  }

  void CalRunningInfo(RnnGenMaskV2TilingParams &tiling_params, int32_t core_num, int32_t block,
                      std::vector<int64_t> & b_shape, std::vector<int64_t> & x_shape)
  {
    OP_LOGD("CalRunningInfo is running");
    int32_t batch_size = x_shape[1];
    int32_t num_step = x_shape[0];
    int32_t hidden_size  = b_shape[0] / 4;
    int32_t rounds = batch_size * num_step;
    int32_t hidden_size_block = ((hidden_size - 1) / block + 1) * block;
    int32_t repeat = hidden_size_block / block;

    tiling_params.cal_mode = CalTilingModeV2(x_shape);
    tiling_params.batch_size = batch_size;
    tiling_params.num_step = num_step;
    tiling_params.hidden_size = hidden_size;
    tiling_params.rounds = rounds;
    tiling_params.hidden_size_block = hidden_size_block;
    tiling_params.repeat = repeat;
    CalCoreInfo(tiling_params, core_num, x_shape);
  }

  void SetRunningInfo(const RnnGenMaskV2TilingParams &tiling_params, OpRunInfo &run_info)
  {
    OP_LOGD("SetRunningInfo is running");
    ByteBufferPut(run_info.tiling_data, tiling_params.cal_mode);
    ByteBufferPut(run_info.tiling_data, tiling_params.core_used);
    ByteBufferPut(run_info.tiling_data, tiling_params.batch_size);
    ByteBufferPut(run_info.tiling_data, tiling_params.num_step);
    ByteBufferPut(run_info.tiling_data, tiling_params.hidden_size);
    ByteBufferPut(run_info.tiling_data, tiling_params.hidden_size_block);
    ByteBufferPut(run_info.tiling_data, tiling_params.repeat);
    ByteBufferPut(run_info.tiling_data, tiling_params.rounds);
    ByteBufferPut(run_info.tiling_data, tiling_params.batch_num_per_aicore);
    ByteBufferPut(run_info.tiling_data, tiling_params.batch_tail);
  }

  void PrintTilingParams(const RnnGenMaskV2TilingParams &tiling_params)
  {
    OP_LOGD("PrintTilingParams is running");
    OP_LOGD("op [RnnGenMaskV2Tiling] : cal_mode=%d.", tiling_params.cal_mode);
    OP_LOGD("op [RnnGenMaskV2Tiling] : core_used=%d.", tiling_params.core_used);
    OP_LOGD("op [RnnGenMaskV2Tiling] : batch_size=%d.", tiling_params.batch_size);
    OP_LOGD("op [RnnGenMaskV2Tiling] : num_step=%d.", tiling_params.num_step);
    OP_LOGD("op [RnnGenMaskV2Tiling] : hidden_size=%d.", tiling_params.hidden_size);
    OP_LOGD("op [RnnGenMaskV2Tiling] : hidden_size_block=%d.", tiling_params.hidden_size_block);
    OP_LOGD("op [RnnGenMaskV2Tiling] : repeat=%d.", tiling_params.repeat);
    OP_LOGD("op [RnnGenMaskV2Tiling] : rounds=%d.", tiling_params.rounds);
    OP_LOGD("op [RnnGenMaskV2Tiling] : batch_num_per_aicore=%d.", tiling_params.batch_num_per_aicore);
    OP_LOGD("op [RnnGenMaskV2Tiling] : batch_tail=%d.", tiling_params.batch_tail);
  }

  bool RnnGenMaskV2Tiling(const std::string &op_type, const TeOpParas &op_paras,
                          const nlohmann::json &op_compile_info, OpRunInfo &run_info)
  {
    OP_LOGD("RnnGenMaskV2Tiling is running");
    int32_t core_num;
    int32_t block;
    bool get_compile_info = GetCompileInfoV2(op_type, op_compile_info, core_num, block);
    if (!get_compile_info)
    {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "RnnGenMaskV2Tiling: GetCompileInfoV2 error.");
      return false;
    }

    RnnGenMaskV2TilingParams tiling_params;
    InitTilingParams(tiling_params);
    std::vector<int64_t> b_shape = op_paras.inputs[1].tensor[0].shape;
    std::vector<int64_t> x_shape = op_paras.inputs[2].tensor[0].shape;
    CalRunningInfo(tiling_params, core_num, block, b_shape, x_shape);
    SetRunningInfo(tiling_params, run_info);
    PrintTilingParams(tiling_params);

    run_info.block_dim = tiling_params.core_used;
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;
    return true;
  }
  REGISTER_OP_TILING_FUNC_BUFFERED(RnnGenMaskV2, RnnGenMaskV2Tiling);
} // namespace optiling.

