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
#include "op_tiling_util.h"
#include "op_log.h"
#include "error_log.h"
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
std::vector<int64_t> setValValue(const std::string& opType, std::vector<std::string> varMap,
                                 const ge::OpDescPtr& op_desc,
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
        } else {
            // elewise var name _dim_x_x when conv+add fusion
            int32_t var_index = 0;
            int32_t dim_index = 0;
            int32_t dim_value = 0;
            if (var.find("dim") != std::string::npos) {
                // dim offset 4 means first 'x' value
                var_index = var.find("dim") + 4;
                OP_TILING_CHECK(var_index > var.length(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "elewise var has no valid dim, var_name:%s", var.c_str()),
                  return varValue);
                try {
                    dim_index = std::stoi(var.substr(var_index, 1).c_str());
                } catch (...) {
                    CUBE_INNER_ERR_REPORT(opType.c_str(), "elewise var has no valid dim_index, var_name:%s", var.c_str());
                    return varValue;
                }
                dim_value = (dim_index == nDim) ? batch : ho*wo;
                varValue.insert(varValue.end(), dim_value);
                GELOGD("elewise fusion optiling var info, var_name: %s, dim_index: %d", var.c_str(), dim_index);
            } else {
                GELOGD("unknow var info, var_name: %s", var.c_str());
            }
        }
    }
    return varValue;
}

// Parse function
template <typename T>
void getCompileInfoList(const nlohmann::json& compileInfo, const std::string& name, std::vector<T>& list)
{
    if (compileInfo.count(name) != 0) {
        for (auto& item: compileInfo[name].items()) {
            list.push_back(item.value().get<T>());
        }
    }
    return;
}

bool getParseInfo(const std::string& opType, const nlohmann::json& opCompileInfo,
    optiling::Conv2DTilingParseInfo& opInfo)
{
    // get fmapC1
    GetCompileValue(opCompileInfo, "fmap_c1", opInfo.fmapC1, 0);
    // get correctRangeFlag
    GetCompileValue(opCompileInfo, "correct_range_flag", opInfo.correctRangeFlag, false);
    // get tilingType
    GetCompileValue(opCompileInfo, "tiling_type", opInfo.tilingType, "");
    // get varMap
    if (opCompileInfo.count("_vars") == 0) {
        GELOGD("Op compile info json doesn't contain _vars info.");
        return false;
    }
    OP_TILING_CHECK(!GetCompileValue(opCompileInfo["_vars"], "10000", opInfo.varMap),
        CUBE_INNER_ERR_REPORT(opType.c_str(), "Conv2DTilingParseFunc, getParseInfo, get varMap error."), return false);
    // get tilingKeyList
    for (auto& item: opCompileInfo["_vars"].items()) {
        opInfo.tilingKeyList.push_back(item.key());
    }
    // get customVarsList
    getCompileInfoList(opCompileInfo, "_custom_vars", opInfo.customVarsList);
    // get defaultRangeList
    getCompileInfoList(opCompileInfo, "default_range", opInfo.defaultRangeList);
    // get tilingRangeList
    getCompileInfoList(opCompileInfo, "tiling_range", opInfo.tilingRangeList);
    // get blockDimList
    if (opCompileInfo.count("block_dim") != 0) {
        for (auto& item: opCompileInfo["block_dim"].items()) {
            opInfo.blockDimList.push_back(item.value().get<int32_t>());
        }
    }
    // get repoSeedsList
    if (opCompileInfo.count("repo_seeds") != 0) {
        for (auto& item: opCompileInfo["repo_seeds"].items()) {
            if (item.value().is_number()) {
                break;
            }
            opInfo.repoSeedsList.push_back(item.value().get<std::vector<int32_t>>());
        }
    }
    // get repoRangeList
    getCompileInfoList(opCompileInfo, "repo_range", opInfo.repoRangeList);
    // get costRangeList
    getCompileInfoList(opCompileInfo, "cost_range", opInfo.costRangeList);
    return true;
}

bool getFuzzyBuildParseInfo(const std::string& opType, const nlohmann::json& opCompileInfo,
    optiling::Conv2DTilingParseInfo& opInfo)
{
    auto& firstOpCompileInfo = opCompileInfo[0];
    OP_TILING_CHECK(!getParseInfo(opType, firstOpCompileInfo, opInfo),
        CUBE_INNER_ERR_REPORT(opType.c_str(),
            "Conv2DTilingParseFunc, getFuzzyBuildParseInfo, get firstOpCompileInfo error."),
        return false);
    for (size_t i = 1; i < opCompileInfo.size(); i++) {
        auto& info = opCompileInfo[i];
        if (info.count("_vars") == 0 ||
            info.count("block_dim") == 0) {
            continue;
        }
        // get tilingKeyList
        opInfo.tilingKeyList.push_back(info["_vars"].begin().key());
        // get blockDimList
        opInfo.blockDimList.push_back(info["block_dim"].begin().value().get<int32_t>());
        // get repoSeedsList
        if (info.contains("repo_seeds") && !info["repo_seeds"].empty()) {
            auto& repoSeeds = info["repo_seeds"].begin().value();
            if (repoSeeds.is_array()) {
                opInfo.repoSeedsList.push_back(repoSeeds.get<std::vector<int32_t>>());
            }
        }
        // get repoRangeList
        if (info.contains("repo_range") && !info["repo_range"].empty()) {
            opInfo.repoRangeList.push_back(info["repo_range"].begin().value().get<std::vector<int64_t>>());
        }
        // get costRangeList
        if (info.contains("cost_range") && !info["cost_range"].empty()) {
            opInfo.costRangeList.push_back(info["cost_range"].begin().value().get<std::vector<int64_t>>());
        }
    }
    return true;
}

bool Conv2DTilingParseFunc(const std::string& opType, const nlohmann::json& opCompileInfo,
    optiling::Conv2DTilingParseInfo& opInfo)
{
    if (opCompileInfo.empty()) {
        GELOGD("op compile info is empty.");
        return false;
    }
    GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());
    // accurate build has only one item
    // fuzzy build has multiple items
    if (opCompileInfo.is_array()) {
        if (!getFuzzyBuildParseInfo(opType, opCompileInfo, opInfo)) {
            GELOGD("Conv2D Tiling, get fuzzy build parse info failed.");
            return false;
        }
    } else if (opCompileInfo.is_object()) {
        if (!getParseInfo(opType, opCompileInfo, opInfo)) {
            GELOGD("Conv2D Tiling, get parse info failed.");
            return false;
        }
    }
    GELOGD("Parse Conv2D CompileInfo successed.");
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

bool Conv2DTiling(const std::string& opType, const ge::Operator& opParas,
    optiling::Conv2DTilingParseInfo& opInfo, utils::OpRunInfo& runInfo)
{
    PROFILING_TILING_INIT(opType.c_str());
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(opParas);
    auto inputDesc = opDesc->GetInputDescPtr(0);
    if (inputDesc == nullptr) {
        OP_LOGE(opType.c_str(), "GetInputDescPtr failed");
	      return false;
    }
    auto outputDesc = opDesc->GetOutputDescPtr(0);
    if (outputDesc == nullptr) {
        OP_LOGE(opType.c_str(), "GetOutputDescPtr failed");
	      return false;
    }
    ge::Format inputFormat = inputDesc->GetFormat();
    std::string xFormat = ge::TypeUtils::FormatToSerialString(inputFormat).c_str();
    if (xFormat != "NC1HWC0" && xFormat != "NHWC") {
        OP_LOGE(opType.c_str(), "only support NC1HWC0 or NHWC format.");
    }

    // default format NC1HWC0
    int32_t nDim = 0;
    int32_t cDim = 1;
    int32_t hDim = 2;
    int32_t wDim = 3;
    if (xFormat == "NHWC") {
        nDim = xFormat.find("N");
        cDim = xFormat.find("C");
        hDim = xFormat.find("H");
        wDim = xFormat.find("W");
    }
    GELOGD("optiling xFormat is %s, nDim = %d, cDim = %d, hDim = %d, wDim = %d",
           xFormat.c_str(), nDim, cDim, hDim, wDim);

    int32_t batch = inputDesc->GetShape().GetDim(nDim);
    int32_t hi = inputDesc->GetShape().GetDim(hDim);
    int32_t wi = inputDesc->GetShape().GetDim(wDim);
    int32_t ho = outputDesc->GetShape().GetDim(hDim);
    int32_t wo = outputDesc->GetShape().GetDim(wDim);

    if (opDesc->GetInputsSize() == 0 ||
        opDesc->GetOutputsSize() == 0 ||
        inputDesc->GetShape().GetDimNum() == 0 ||
        outputDesc->GetShape().GetDimNum() == 0) {
        OP_LOGE(opType.c_str(), "inputsize or outputsize is zero");
        return false;
    }
    if (ho != 1 && wo == 1) {
        OP_LOGE(opType.c_str(), "not support ho != 1 and wo == 1.");
    }
    PROFILING_TILING_AFTER_GET_SHAPE_REG();
    if (opType.compare("Conv2D") == 0 && opInfo.fmapC1 != 0 &&
        inputDesc->GetShape().GetDim(cDim) != opInfo.fmapC1) {
        CUBE_INNER_ERR_REPORT(opType.c_str(), "Not support, input x channel should be equal to filter channel*groups;"
                              "x_channel=%d, fmap_c1=%d", (int32_t)inputDesc->GetShape().GetDim(cDim),
                              opInfo.fmapC1);
        return false;
    }
    PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
    std::vector<int64_t> varValue = setValValue(opType, opInfo.varMap, opDesc, nDim, hDim, wDim);
    bool res = cube_tiling1_v4(opType, inputDesc->GetShape().GetDims(), xFormat, varValue, opInfo, runInfo);
    PROFILING_TILING_AFTER_CALCU_TILING_REG();
    std::string nodeName = opDesc->GetName();
    GELOGD("[%s] tiling_data is %d, %d, %d, %d, %d, %d", nodeName.c_str(),
           runInfo.GetTilingKey(), batch, hi, ho, wi, wo);
    PROFILING_TILING_END();
    return res;
}

// register tiling interface of the conv2d
REGISTER_OP_TILING_V4_CUSTOM(Conv2D, Conv2DTiling, Conv2DTilingParseFunc, Conv2DTilingParseInfo);
REGISTER_OP_TILING_V4_CUSTOM(DepthwiseConv2D, Conv2DTiling, Conv2DTilingParseFunc, Conv2DTilingParseInfo);
}  // namespace optiling
