/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file  batch_multi_class_non_max_suppression.cc
 * \brief
 */
#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {
  struct BatchMultiClassNonMaxSuppressionTilingParams {
    int32_t cal_mode;
    int32_t core_used;
    int32_t batch_per_core;
    int32_t batch_last_core;
    int32_t batch;
    int32_t classes;
    int32_t boxes_num;
    int32_t topk_loop_time;
    int32_t topk_loop_tail;
  };

  static void InitTilingParams(BatchMultiClassNonMaxSuppressionTilingParams &params) {
    OP_LOGD("InitTilingParams is running");
    params.cal_mode = 0;
    params.core_used = 0;
    params.batch_per_core = 0;
    params.batch_last_core = 0;
    params.batch = 0;
    params.classes = 0;
    params.boxes_num = 0;
    params.topk_loop_time = 0;
    params.topk_loop_tail = 0;
  }

  static bool GetCompileInfo(const std::string &op_type, const nlohmann::json &op_compile_info,
                             int32_t &aicore_num, int32_t &proposal_topk_k) {
    OP_LOGD("GetCompileInfo is running");
    using namespace nlohmann;
    auto all_vars = op_compile_info["vars"];
    if (all_vars.count("aicore_num") == 0) {
      OP_LOGE("op [BatchMultiClassNonMaxSuppressionTiling] : GetCompileInfo, get total_core_num error");
      return false;
    }
    aicore_num = all_vars["aicore_num"].get<std::int32_t>();
    if (all_vars.count("proposal_topk_k") == 0) {
      OP_LOGE("op [BatchMultiClassNonMaxSuppressionTiling] : GetCompileInfo, get total_core_num error");
      return false;
    }
    proposal_topk_k = all_vars["proposal_topk_k"].get<std::int32_t>();
    return true;
  }

  static int32_t CalTilingMode(std::vector<int64_t> scores_shape) {
    OP_LOGD("CalTilingMode is running");
    int32_t tiling_mode = 0;
    auto boxes_num = scores_shape[1];
    if (boxes_num >= 1) {
      tiling_mode = 1;
    }
    return tiling_mode;
  }

  static void CalCoreInfo(BatchMultiClassNonMaxSuppressionTilingParams &tiling_params,
                          int32_t & core_num, std::vector<int64_t> & scores_shape) {
    OP_LOGD("CalCoreInfo is running");
    OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING("batch_multi_class_non_max_suppression",
      "core_num = 0 is not support"), return);
    int32_t batch = scores_shape[0];
    int32_t batch_per_core = 0;
    int32_t core_used = 0;
    int32_t batch_last_core = 0;

    batch_per_core = (batch - 1) / core_num + 1;
    core_used = (batch - 1) / batch_per_core + 1;
    batch_last_core = batch - (core_used - 1) * batch_per_core;
    tiling_params.batch_per_core = batch_per_core;
    tiling_params.core_used = core_used;
    tiling_params.batch_last_core = batch_last_core;
  }

  static void CalRunningInfo(BatchMultiClassNonMaxSuppressionTilingParams &tiling_params,
                             int32_t core_num, int32_t proposal_topk_k, std::vector<int64_t> & scores_shape) {
    OP_LOGD("CalRunningInfo is running");
    OP_TILING_CHECK(proposal_topk_k == 0, VECTOR_INNER_ERR_REPORT_TILIING("batch_multi_class_non_max_suppression",
      "proposal_topk_k = 0 is not support"), return);
    int32_t batch = scores_shape[0];
    int32_t classes = scores_shape[1];
    int32_t boxes_num = scores_shape[2];
    int32_t topk_loop_time = boxes_num / proposal_topk_k;
    int32_t topk_loop_tail = boxes_num % proposal_topk_k;

    tiling_params.cal_mode = CalTilingMode(scores_shape);
    tiling_params.batch = batch;
    tiling_params.classes = classes;
    tiling_params.boxes_num = boxes_num;
    tiling_params.topk_loop_time = topk_loop_time;
    tiling_params.topk_loop_tail = topk_loop_tail;
    CalCoreInfo(tiling_params, core_num, scores_shape);
  }

  static void SetRunningInfo(const BatchMultiClassNonMaxSuppressionTilingParams &tiling_params, OpRunInfo &run_info) {
    OP_LOGD("SetRunningInfo is running");
    ByteBufferPut(run_info.tiling_data, tiling_params.cal_mode);
    ByteBufferPut(run_info.tiling_data, tiling_params.core_used);
    ByteBufferPut(run_info.tiling_data, tiling_params.batch_per_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.batch_last_core);
    ByteBufferPut(run_info.tiling_data, tiling_params.batch);
    ByteBufferPut(run_info.tiling_data, tiling_params.classes);
    ByteBufferPut(run_info.tiling_data, tiling_params.boxes_num);
    ByteBufferPut(run_info.tiling_data, tiling_params.topk_loop_time);
    ByteBufferPut(run_info.tiling_data, tiling_params.topk_loop_tail);
  }

  static void PrintTilingParams(const BatchMultiClassNonMaxSuppressionTilingParams &tiling_params) {
    OP_LOGD("PrintTilingParams is running");
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : cal_mode=%d.", tiling_params.cal_mode);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : core_used=%d.", tiling_params.core_used);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : batch_per_core=%d.", tiling_params.batch_per_core);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : batch_last_core=%d.", tiling_params.batch_last_core);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : batch=%d.", tiling_params.batch);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : classes=%d.", tiling_params.classes);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : boxes_num=%d.", tiling_params.boxes_num);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : topk_loop_time=%d.", tiling_params.topk_loop_time);
    OP_LOGD("op [BatchMultiClassNonMaxSuppressionTiling] : topk_loop_tail=%d.", tiling_params.topk_loop_tail);
  }

  bool BatchMultiClassNonMaxSuppressionTiling(const std::string &op_type, const TeOpParas &op_paras,
                                              const nlohmann::json &op_compile_info, OpRunInfo &run_info) {
    OP_LOGD("BatchMultiClassNonMaxSuppressionTiling is running");
    using namespace ge;
    int32_t core_num;
    int32_t proposal_topk_k;
    bool get_compile_info = GetCompileInfo(op_type, op_compile_info, core_num, proposal_topk_k);
    if (!get_compile_info) {
      OP_LOGE("op[%s] BatchMultiClassNonMaxSuppressionTiling: GetCompileInfo error.", op_type.c_str());
      return false;
    }

    BatchMultiClassNonMaxSuppressionTilingParams tiling_params;
    InitTilingParams(tiling_params);
    std::vector<int64_t> scores_shape = op_paras.inputs[1].tensor[0].shape;
    CalRunningInfo(tiling_params, core_num, proposal_topk_k, scores_shape);
    SetRunningInfo(tiling_params, run_info);
    PrintTilingParams(tiling_params);

    run_info.block_dim = tiling_params.core_used;
    std::vector<int64_t> workspace;
    run_info.workspaces = workspace;
    return true;
  }
  REGISTER_OP_TILING_FUNC_BUFFERED(BatchMultiClassNonMaxSuppression, BatchMultiClassNonMaxSuppressionTiling);
} // namespace optiling.
