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
 * \file prod_env_mat_a_calc_descrpt.cc
 * \brief tiling function of op
 */
#include <string>
#include <nlohmann/json.hpp>
#include "../op_proto/util/error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "error_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "op_log.h"

namespace optiling {
/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] opCompileInfo: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool ProdEnvMatACalcDescrptTiling(const std::string& opType, const TeOpParas& opParas,
                                  const nlohmann::json& opCompileInfo, OpRunInfo& runInfo) {
  OP_LOGD(opType.c_str(), "ProdEnvMatACalcDescrptTiling run begin");
  int32_t blockDim = 8;
  runInfo.block_dim = blockDim;

  OP_LOGD(opType.c_str(), "ProdEnvMatACalcDescrptTiling run success");
  return true;
}
// register tiling interface of the ProdEnvMatACalcDescrpt op.
REGISTER_OP_TILING_FUNC_BUFFERED(ProdEnvMatACalcDescrpt, ProdEnvMatACalcDescrptTiling);
}  // namespace optiling
