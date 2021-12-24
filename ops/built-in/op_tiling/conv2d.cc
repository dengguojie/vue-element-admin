/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file conv2d.cpp
 * \brief tiling function of conv2d
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "cube_tiling.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "external/graph/operator.h"
#include "op_tiling.h"
#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

using namespace std;
namespace optiling {
/*
 * @brief: set val value
 * @param [in] varMap: varMap of conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [out] valValue: val value
 */
std::vector<int64_t> setValValue(std::vector<std::string> varMap, const ge::OpDescPtr& op_desc,
                                 int32_t nDim, int32_t hDim, int32_t wDim) {
    auto input_desc = op_desc->GetInputDescPtr(0);
    auto output_desc = op_desc->GetOutputDescPtr(0);

    int32_t batch = input_desc->GetShape().GetDim(nDim);
    int32_t hi = input_desc->GetShape().GetDim(hDim);
    int32_t wi = input_desc->GetShape().GetDim(wDim);
    int32_t ho = output_desc->GetShape().GetDim(hDim);
    int32_t wo = output_desc->GetShape().GetDim(wDim);
    GELOGD("optiling runing shape is %d, %d, %d, %d, %d", batch, hi, ho, wi, wo);

    std::vector<int64_t> varValue;
    for (auto var : varMap) {
        if (var == "batch_n") {
            varValue.insert(varValue.end(), batch);
        } else if (var == "fmap_h") {
            varValue.insert(varValue.end(), hi);
        } else if (var == "fmap_w") {
            varValue.insert(varValue.end(), wi);
        } else if (var == "ho") {
            varValue.insert(varValue.end(), ho);
        } else if (var == "wo") {
            varValue.insert(varValue.end(), wo);
        }
    }
    return varValue;
}

/*
 * @brief: get op infos for fuzzy build, which has multiple items
 * @param [in] opCompileInfo: compile time generated info of the conv2d
 * @param [out] varMap: value map
 * @param [out] opInfo: op info
 * @return bool: success or not
 */
bool getFuzzyBuildInfo(const nlohmann::json& opCompileInfo, 
                       std::vector<std::string>& varMap, 
                       nlohmann::json& opInfo) {
    // >>> start: splice compile info
    opInfo = opCompileInfo[0];
    try {
        varMap = opInfo.at("_vars").begin().value().get<std::vector<std::string>>();
    }catch (nlohmann::json::out_of_range& e) {
        return false;
    }
    nlohmann::json item;
    for (size_t i = 1; i < opCompileInfo.size(); ++i) {
        item = opCompileInfo[i];
        std::vector<std::string> key_list = {"repo_seeds", 
		                             "repo_range", 
                                             "cost_range"};
        for (auto key: key_list) {
            if (item[key].is_object() && !item[key].empty()) {
                std::vector<int32_t> list_value = item[key].begin().value().get<std::vector<int32_t>>();
                opInfo[key][item[key].begin().key()] = list_value;
            }
        }
        std::vector<std::string> key_int = {"block_dim"};
        for (auto key: key_int) {
            if (item[key].is_object() && !item[key].empty()) {
                int32_t int_value = item[key].begin().value().get<int32_t>();
                opInfo[key][item[key].begin().key()] = int_value;
            }
        }
    }
    // <<< end: put together compile info
    return true;
}

/*
 * @brief: tiling function of conv2d
 * @param [in] op_type: op_type of the conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [in] op_compile_info: compile time generated info of the conv2d
 * @param [out] run_info: result data
 * @return bool: success or not
 */

bool Conv2DTiling(const std::string& opType, 
		  const ge::Operator& opParas, 
                  const nlohmann::json& opCompileInfo, 
                  utils::OpRunInfo& runInfo) {
    PROFILING_TILING_INIT(opType.c_str());
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(opParas);
    auto input_desc = op_desc->GetInputDescPtr(0);
    if (input_desc == nullptr) {
        OP_LOGE(opType.c_str(), "GetInputDescPtr failed");
	return false;
    }
    auto output_desc = op_desc->GetOutputDescPtr(0);
    if (output_desc == nullptr) {
        OP_LOGE(opType.c_str(), "GetOutputDescPtr failed");
	return false;
    }
    ge::Format input_format = input_desc->GetFormat();
    std::string x_format = ge::TypeUtils::FormatToSerialString(input_format).c_str();
    if (x_format != "NC1HWC0" && x_format != "NHWC") {
        OP_LOGE(opType.c_str(), "only support NC1HWC0 or NHWC format.");
    }

    // default format NC1HWC0
    int32_t nDim = 0;
    int32_t cDim = 1;
    int32_t hDim = 2;
    int32_t wDim = 3;
    if (x_format == "NHWC") {
        nDim = x_format.find("N");
        cDim = x_format.find("C");
        hDim = x_format.find("H");
        wDim = x_format.find("W");
    }
    GELOGD("optiling x_format is %s, nDim = %d, cDim = %d, hDim = %d, wDim = %d", 
           x_format.c_str(), nDim, cDim, hDim, wDim);

    int32_t batch = input_desc->GetShape().GetDim(nDim);
    int32_t hi = input_desc->GetShape().GetDim(hDim);
    int32_t wi = input_desc->GetShape().GetDim(wDim);
    int32_t ho = output_desc->GetShape().GetDim(hDim);
    int32_t wo = output_desc->GetShape().GetDim(wDim);


    if (op_desc->GetInputsSize() == 0 || 
        op_desc->GetOutputsSize() == 0 ||
        input_desc->GetShape().GetDimNum() == 0 ||
       	output_desc->GetShape().GetDimNum() == 0) {
        OP_LOGE(opType.c_str(), "inputsize or outputsize is zero");
        return false;
    }

    if (opType.c_str() == "Conv2D" && opCompileInfo.contains("fmap_c1") &&
        input_desc->GetShape().GetDim(cDim) != opCompileInfo["fmap_c1"]) {
        CUBE_INNER_ERR_REPORT(opType.c_str(), "Not support, input x channel should be equal to filter channel*groups;"
                              "x_channel=%d, fmap_c1=%d", (int32_t)input_desc->GetShape().GetDim(cDim),
                              (int32_t)opCompileInfo["fmap_c1"]);
        return false;
    }

    if (opCompileInfo.empty()) {
        GELOGD("op compile info is empty");
        return false;
    }

    if (ho != 1 && wo == 1) {
        OP_LOGE(opType.c_str(), "not support ho != 1 and wo == 1.");
    }

    // accurate build has only one item
    // fuzzy build has multiple items
    PROFILING_TILING_AFTER_GET_SHAPE_REG()
    std::vector<std::string> varMap;
    nlohmann::json opInfo;
    GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());
    if (opCompileInfo.is_array()) {
        if (!getFuzzyBuildInfo(opCompileInfo, varMap, opInfo)) {
            return false;
        }
        GELOGD("compile info after splice is: %s", opInfo.dump().c_str());
    } else if (opCompileInfo.is_object()) {
        varMap = opCompileInfo.at("_vars")["10000"].get<std::vector<std::string>>();
        opInfo = opCompileInfo;
    }
    PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG()
    std::vector<int64_t> varValue = setValValue(varMap, 
		                                op_desc, 
						nDim, 
                                                hDim, 
						wDim);
    bool res = cube_tiling1(opType, 
		            input_desc->GetShape().GetDims(),
                            x_format, 
			    varValue, 
			    opInfo, 
			    runInfo);
    PROFILING_TILING_AFTER_CALCU_TILING_REG()
    
    std::string node_name = op_desc->GetName();
    GELOGD("[%s] tiling_data is %d, %d, %d, %d, %d, %d", node_name.c_str(),
           runInfo.GetTilingKey(), batch, hi, ho, wi, wo);

    PROFILING_TILING_END()
    return res;
}

// register tiling interface of the conv2d
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv2D, Conv2DTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(DepthwiseConv2D, Conv2DTiling);
}  // namespace optiling
