/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file quant_host_cpu_op_common.h
 * \brief fe_util
 */
#ifndef BUILTIN_FUSIONPASS_QUANT_COMMON_H
#define BUILTIN_FUSIONPASS_QUANT_COMMON_H

#include <string>
#include <vector>
#include "external/graph/types.h"
#include "graph/compute_graph.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
const int32_t INDEX_CO = 0;
const int32_t INDEX_CI = 1;
const int32_t INDEX_FILTER_H = 2;
const int32_t INDEX_FILTER_W = 3;
const int32_t LAST_AXIS_INDEX = 3;

/* Attribute Name */
const std::string QUANT_SCALE = "quant_scale";
const std::string DEQUANT_SCALE = "dequant_scale";
const std::string ATTR_OFFSET_X = "offset_x";
const std::string ATTR_OFFSET = "offset";
const std::string ATTR_SCALE = "scale";

fe::Status GetkernelDataCountForPass(const std::vector<int64_t>& filterDIms, int64_t& kernelDataCount);
}  // namespace fe
#endif  // BUILTIN_FUSIONPASS_QUANT_COMMON_H