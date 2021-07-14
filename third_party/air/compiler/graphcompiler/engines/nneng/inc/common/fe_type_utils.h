/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_FE_TYPE_UTILS_H_
#define FUSION_ENGINE_INC_COMMON_FE_TYPE_UTILS_H_

#include "common/aicore_util_types.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
std::string GetRealPath(const std::string &path);
// send error message to error manager
void ReportErrorMessage(std::string error_code, const std::map<std::string, std::string> &args_map);
Status String2DataType(std::string dtype_str, ge::DataType &dtype);
std::string GetStrByFormatVec(const std::vector<ge::Format>& format_vec);
std::string GetStrByDataTypeVec(const std::vector<ge::DataType>& data_type_vec);

std::string GetOpPatternString(OpPattern op_pattern);

std::string GetPrecisionPolicyString(PrecisionPolicy precision_policy);

bool IsMemoryEmpty(const ge::GeTensorDesc &tensor_desc);

}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_FE_TYPE_UTILS_H_

