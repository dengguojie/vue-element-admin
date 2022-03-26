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
 * \file resize.cc
 * \brief
 */
#include "resize_common.h"


namespace optiling {
bool ResizeParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                     ResizeCommonInputCompile& compile_value) {
    const int64_t nearest_2d = 20;
    const int64_t linear_2d = 21;
    if (compile_info.count("vars") == 0) {
        return false;
    }
    int64_t mode_name;
    const nlohmann::json& all_vars = compile_info["vars"];
    OP_TILING_CHECK(!GetCompileValue(all_vars, "mode_name", mode_name),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeParseFunc, get mode_name error"),
                    return false);
    std::string op_type_mode;
    if (mode_name == nearest_2d) {
        op_type_mode = "ResizeNearestNeighborV2";
    } else if (mode_name == linear_2d) {
        op_type_mode = "ResizeBilinearV2";
    } else {
        OP_LOGE(op_type.c_str(), "Mode only support nearest or linear.");
        return false;
    }
    return ResizeCommonParseFunc(op_type_mode, compile_info, compile_value);
}
bool ResizeTiling(const std::string& op_type, const ge::Operator& op_paras,
                  const ResizeCommonInputCompile& op_info, utils::OpRunInfo& run_info) {
    using namespace ge;
    ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    OP_TILING_CHECK(op_desc == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The op_desc is nullptr."),
                    return false);
    std::string mode = "nearest";
    ge::AttrUtils::GetStr(op_desc, "mode", mode);

    ge::Format input_format = op_paras.GetInputDesc(0).GetFormat();
    if (input_format == FORMAT_NC1HWC0) {
      std::string op_type_2d;
      if (mode == "nearest") {
        op_type_2d = "ResizeNearestNeighborV2";
      } else if (mode == "linear") {
        op_type_2d = "ResizeBilinearV2";
      } else {
        OP_LOGE(op_type.c_str(), "Mode only support nearest or linear.");
        return false;
      }
      return ResizeCommonTiling(op_type_2d, op_paras, op_info, run_info);
    } else {
      OP_LOGE(op_type.c_str(), "Only support 2D resize.");
      return false;
    }
}
// register tiling interface of the Resize op.
REGISTER_OP_TILING_V3_CUSTOM(Resize, ResizeTiling, ResizeParseFunc, ResizeCommonInputCompile);
}  // namespace optiling.