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
#ifndef FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H
#define FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H

#include "graph/compute_graph.h"
#include "aicore_util_types.h"
#include "op_slice_info.h"

namespace fe {
    void SetOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_s);

    void GetOpSliceInfoFromJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

    void SetFusionOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_s);
} // namespace fe
#endif // FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H