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
#include "conv2d.h"
#include <map>
#include <vector>
#include <string>
#include <bitset>
#include <cmath>

using namespace std;
namespace optiling {
/*
 * @brief: set val value
 * @param [in] varMap: varMap of conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [out] valValue: val value
 */
std::vector<int64_t> setValValue(const std::string& opType, std::vector<std::string> varMap,
                                 const ge::OpDescPtr& opDesc,
                                 int32_t nDim, int32_t hDim, int32_t wDim) {
    auto inputDesc = opDesc->GetInputDescPtr(0);
    auto outputDesc = opDesc->GetOutputDescPtr(0);

    int32_t batch = inputDesc->GetShape().GetDim(nDim);
    int32_t hi = inputDesc->GetShape().GetDim(hDim);
    int32_t wi = inputDesc->GetShape().GetDim(wDim);
    int32_t ho = outputDesc->GetShape().GetDim(hDim);
    int32_t wo = outputDesc->GetShape().GetDim(wDim);
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
                OP_TILING_CHECK(var_index > static_cast<int32_t>(var.length()),
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

// Parse functions
bool GetHardwareInfo(const std::string& opType, const nlohmann::json& compileInfo, const std::string& name,
    optiling::Conv2DTilingParseInfo& opInfo)
{
    const nlohmann::json& hardwareInfo = compileInfo[name];
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "aicore_num", opInfo.aicoreNum), false, opType, "Get aicoreNum failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l2_size", opInfo.l2Size), false, opType, "Get l2Size failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l1_size", opInfo.l1Size), false, opType, "Get l1Size failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l0_a_size", opInfo.l0aSize), false, opType, "Get l0aSize failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l0_b_size", opInfo.l0bSize), false, opType, "Get l0bSize failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l0_c_size", opInfo.l0cSize), false, opType, "Get l0cSize failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "ub_size", opInfo.ubSize), false, opType, "Get ubSize failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "bt_size", opInfo.btSize), false, opType, "Get btSize failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "ddr_read_rate", opInfo.ddrReadRate),
        false, opType, "Get ddrReadRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "ddr_write_rate", opInfo.ddrWriteRate),
        false, opType, "Get ddrWriteRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l2_rate", opInfo.l2Rate), false, opType, "Get l2Rate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l2_read_rate", opInfo.l2ReadRate),
        false, opType, "Get l2ReadRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l2_write_rate", opInfo.l2WriteRate),
        false, opType, "Get l2WriteRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l1_to_l0_a_rate", opInfo.l1ToL0aRate),
        false, opType, "Get l1ToL0aRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l1_to_l0_b_rate", opInfo.l1ToL0bRate),
        false, opType, "Get l1ToL0bRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l1_to_ub_rate", opInfo.l1ToUbRate),
        false, opType, "Get l1ToUbRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "l0_c_to_ub_rate", opInfo.l0cToUbRate),
        false, opType, "Get l0cToUbRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "ub_to_l2_rate", opInfo.ubToL2Rate),
        false, opType, "Get ubToL2Rate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "ub_to_ddr_rate", opInfo.ubToDdrRate),
        false, opType, "Get ubToDdrRate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "ub_to_l1_rate", opInfo.ubToL1Rate),
        false, opType, "Get ubToL1Rate failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "cube_bandwidth", opInfo.cubeBandwidth),
        false, opType, "Get cubeBandwidth failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "vector_bandwidth", opInfo.vectorBandwidth),
        false, opType, "Get vectorBandwidth failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "cube_vector_split_bool", opInfo.cubeVectorSplit),
        false, opType, "Get cubeVectorSplit failed!");
    OP_LOGE_IF(!GetCompileValue(hardwareInfo, "soc_version", opInfo.socVersion),
        false, opType, "Get socVersion failed!");
    return true;
}

bool GetFusionUtilize(const std::string& opType, const nlohmann::json& compileInfo, const std::string& name,
    optiling::Conv2DTilingParseInfo& opInfo)
{
    const nlohmann::json& fusionUtilize = compileInfo[name];
    OP_LOGE_IF(!GetCompileValue(fusionUtilize, "pre_fusion_ub_utilize", opInfo.preFusionUbUtilize),
        false, opType, "Get preFusionUbUtilize failed!");
    OP_LOGE_IF(!GetCompileValue(fusionUtilize, "post_fusion_ub_utilize", opInfo.postFusionUbUtilize),
        false, opType, "Get postFusionUbUtilize failed!");
    OP_LOGE_IF(!GetCompileValue(fusionUtilize, "pre_fusion_vector_utilize", opInfo.preFusionVectorUtilize),
        false, opType, "Get preFusionVectorUtilize failed!");
    OP_LOGE_IF(!GetCompileValue(fusionUtilize, "post_fusion_vector_utilize", opInfo.postFusionVectorUtilize),
        false, opType, "Get postFusionVectorUtilize failed!");
    return true;
}

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
    // get hardware info and ub/vector fusion utilize
    OP_LOGE_IF(!GetHardwareInfo(opType, opCompileInfo, "hardware_info", opInfo), false, opType, "Get hardware failed!");
    OP_LOGE_IF(!GetFusionUtilize(opType, opCompileInfo, "fusion_utilize", opInfo), false, opType, "Get fusion failed!");
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
    const auto& firstOpCompileInfo = opCompileInfo[0];
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

uint32_t Lcm(const uint32_t valueA, const uint32_t valueB)
{
    if (valueB == 0 || valueA == 0) {
        GELOGD("The denominator cannot be zore!");
    }
    uint32_t para1 = valueA;
    uint32_t para2 = valueB;
    uint32_t tmpValue = para1;
    while (para1 % para2 != 0) {
        tmpValue = para1;
        para1 = para2;
        para2 = tmpValue % para2;
    }

    return (valueA * valueB) / para2;
}

uint32_t Conv2dBinaryTiling::GetMKN(ge::DataType dType, uint32_t idx)
{
    if (CUBE_MKN_MAP.find(dType) != CUBE_MKN_MAP.end() && idx < CUBE_MKN_MAP[dType].size()) {
        return CUBE_MKN_MAP[dType][idx];
    }
    GELOGD("[%s] Unexcept dtype or index for CUBE_MKN_MAP, please check input!", nodeName.c_str());
    return MKN_VALUE_DEFAULT;
}


/*
 * @brief: check range
 */
inline bool Conv2dBinaryTiling::CheckRange(int32_t value, int32_t lowerLimit, int32_t upperLimit)
{
    return value >= lowerLimit and value <= upperLimit;
}

bool Conv2dBinaryTiling::InitConvUbUtilize(const optiling::Conv2DTilingParseInfo& opInfo)
{
    // init ub utilize and vector utilize for fusion
    convParas.preFusionUbUtilize = opInfo.preFusionUbUtilize;
    convParas.preFusionVectorUtilize = opInfo.preFusionVectorUtilize;
    convParas.postFusionUbUtilize = opInfo.postFusionUbUtilize;
    convParas.postFusionVectorUtilize = opInfo.preFusionVectorUtilize;
    GELOGD("Conv2dBinaryTiling fusion utilize info: "
        "preFusionUbUtilize=%.3f, preFusionVectorUtilize=%d, postFusionUbUtilize=%.3f, postFusionVectorUtilize=%d",
        convParas.preFusionUbUtilize, convParas.preFusionVectorUtilize,
        convParas.postFusionUbUtilize, convParas.postFusionVectorUtilize);

    return true;
}

bool Conv2dBinaryTiling::InitHardwareInfo(const optiling::Conv2DTilingParseInfo& opInfo)
{
    hardwareInfo.aicoreNum = opInfo.aicoreNum;
    hardwareInfo.l2Size = opInfo.l2Size;
    hardwareInfo.l1Size = opInfo.l1Size;
    hardwareInfo.l0aSize = opInfo.l0aSize;
    hardwareInfo.l0bSize = opInfo.l0bSize;
    hardwareInfo.l0cSize = opInfo.l0cSize;
    hardwareInfo.ubSize = opInfo.ubSize;
    hardwareInfo.btSize = opInfo.btSize;
    hardwareInfo.ddrReadRate = opInfo.ddrReadRate;
    hardwareInfo.ddrWriteRate = opInfo.ddrWriteRate;
    hardwareInfo.l2Rate = opInfo.l2Rate;
    hardwareInfo.l2ReadRate = opInfo.l2ReadRate;
    hardwareInfo.l2WriteRate = opInfo.l2WriteRate;
    hardwareInfo.l1ToL0aRate = opInfo.l1ToL0aRate;
    hardwareInfo.l1ToL0bRate = opInfo.l1ToL0aRate;
    hardwareInfo.l1ToUbRate = opInfo.l1ToUbRate;
    hardwareInfo.l0cToUbRate = opInfo.l0cToUbRate;
    hardwareInfo.ubToL2Rate = opInfo.ubToL2Rate;
    hardwareInfo.ubToDdrRate = opInfo.ubToDdrRate;
    hardwareInfo.ubToL1Rate = opInfo.ubToL1Rate;
    hardwareInfo.cubeBandwidth = opInfo.cubeBandwidth;
    hardwareInfo.vectorBandwidth = opInfo.vectorBandwidth;
    hardwareInfo.cubeVectorSplit = opInfo.cubeVectorSplit;
    hardwareInfo.socVersion = opInfo.socVersion;

    GELOGD("Conv2dBinaryTiling hardware info: "
        "aicoreNum=%d, l2Size=%d, l1Size=%d, l0aSize=%d, l0bSize=%d, "
        "l0cSize=%d, ubSize=%d, btSize=%d, ddrReadRate=%d, "
        "ddrWriteRate=%d, l2Rate=%d, l2ReadRate=%d, l2WriteRate=%d, "
        "l1ToL0aRate=%d, l1ToL0bRate=%d, l1ToUbRate=%d, l0cToUbRate=%d, "
        "ubToL2Rate=%d, ubToDdrRate=%d, ubToL1Rate=%d, cubeBandwidth=%d, "
        "vectorBandwidth=%d, cubeVectorSplit=%d, socVersion=%s",
        hardwareInfo.aicoreNum, hardwareInfo.l2Size, hardwareInfo.l1Size, hardwareInfo.l0aSize, hardwareInfo.l0bSize,
        hardwareInfo.l0cSize, hardwareInfo.ubSize, hardwareInfo.btSize, hardwareInfo.ddrReadRate,
        hardwareInfo.ddrWriteRate, hardwareInfo.l2Rate, hardwareInfo.l2ReadRate, hardwareInfo.l2WriteRate,
        hardwareInfo.l1ToL0aRate, hardwareInfo.l1ToL0bRate, hardwareInfo.l1ToUbRate, hardwareInfo.l0cToUbRate,
        hardwareInfo.ubToL2Rate, hardwareInfo.ubToDdrRate, hardwareInfo.ubToL1Rate, hardwareInfo.cubeBandwidth,
        hardwareInfo.vectorBandwidth, hardwareInfo.cubeVectorSplit, hardwareInfo.socVersion.c_str());

    return true;
}

bool Conv2dBinaryTiling::CheckL1SizeBound() {
    uint32_t hoNum = GetMKN(convParas.bType, 0) / convParas.wo + 2;
    uint32_t hkDilation = (convParas.kh - 1) * convParas.dilations_h + 1;
    float maxFmapL1 = ((hoNum - 1) * convParas.stride_h + hkDilation) * convParas.wi * \
                           GetMKN(convParas.aType, 1) * M_BIT_RATIO[convParas.aType];

    return maxFmapL1 > L1_BUFFER_SIZE ? true : false;
}

/*
 * @brief: parser convparams from opDesc
 * @param [in] opDesc: input/output desc
 * @param [in] opInfo: contains harewareinfo or compileinfo
 */
bool Conv2dBinaryTiling::ParserConv2DParas(const ge::OpDescPtr& opDesc, const optiling::Conv2DTilingParseInfo& opInfo)
{
    std::vector<int32_t> padList;
    std::vector<int32_t> dilationList;
    std::vector<int32_t> stridesList;
    opType = opDesc->GetType();
    nodeName = opDesc->GetName();
    ge::ConstGeTensorDescPtr inputDesc = opDesc->GetInputDescPtr(0);
    ge::ConstGeTensorDescPtr filterDesc = opDesc->GetInputDescPtr(1);
    ge::ConstGeTensorDescPtr biasDesc = opDesc->GetInputDescPtr(2);
    ge::ConstGeTensorDescPtr outputDesc = opDesc->GetOutputDescPtr(0);

    OP_LOGE_IF(inputDesc->GetFormat() != ge::Format::FORMAT_NC1HWC0, false, opType,
        "input0 only support NC1HWC0 format!");
    OP_LOGE_IF(filterDesc->GetFormat() != ge::Format::FORMAT_FRACTAL_Z, false, opType,
        "filter only support FRACZ format!");
    OP_LOGE_IF(outputDesc->GetFormat() != ge::Format::FORMAT_NC1HWC0, false, opType,
        "output only support NC1HWC0 format!");

    OP_LOGE_IF(inputDesc->GetShape().GetDimNum() != kNC1HWC0DimSize, false, opType,
        "input0 shape only support 5HD!");
    OP_LOGE_IF(filterDesc->GetShape().GetDimNum() != kFRACZDimSize || \
        filterDesc->GetOriginShape().GetDimNum() != kNCHWDimSize, false, opType,
        "input1 shape only support FRACZ and ori_shape 4D!");
    OP_LOGE_IF(outputDesc->GetShape().GetDimNum() != kNC1HWC0DimSize, false, opType,
        "output shape only support 5D!");

    uint32_t cinOri = 0;
    uint32_t coutOri = 0;
    uint32_t c0Val = GetMKN(convParas.bType, 1);
    uint32_t cout0 = GetMKN(convParas.bType, MKN_NINDEX);
    uint32_t groupsOri = 1;
    OP_LOGE_IF(!ge::AttrUtils::GetInt(opDesc, "groups", groupsOri), false, opType,
        "get attr groups desc failed!");
    if (inputDesc->GetOriginFormat() == ge::Format::FORMAT_NCHW) {
        cinOri = inputDesc->GetOriginShape().GetDim(kCDimNCHWIdx) / groupsOri;
    } else {
        cinOri = inputDesc->GetOriginShape().GetDim(kCDimNHWCIdx) / groupsOri;
    }
    coutOri = filterDesc->GetOriginShape().GetDim(kNDimNHWCIdx) / groupsOri;
    int32_t enlarge = min(Lcm(Lcm(cinOri, c0Val) / cinOri, Lcm(coutOri, cout0) / coutOri), groupsOri);
    uint32_t c1Opt = ceil((float)(cinOri * enlarge) / GetMKN(convParas.bType, 1));
    uint32_t cout1Opt = ceil((float)(coutOri * enlarge) / GetMKN(convParas.bType, 2));
    uint32_t groupOpt = ceil((float)groupsOri / max(enlarge, 1));
    GELOGD("[%s] Get group opt success, enlarge = %d, c1Opt = %d, cout1Opt = %d, groupOpt = %d", \
           nodeName.c_str(), enlarge, c1Opt, cout1Opt, groupOpt);

    convParas.groups = groupOpt;
    convParas.batch = inputDesc->GetShape().GetDim(kNDimNC1HWC0Idx);
    convParas.fmci = c1Opt * c0Val;
    convParas.hi = inputDesc->GetShape().GetDim(kHDimNC1HWC0Idx);
    convParas.wi = inputDesc->GetShape().GetDim(kWDimNC1HWC0Idx);
    if (filterDesc->GetOriginFormat() == ge::Format::FORMAT_NCHW) {
        convParas.kh = filterDesc->GetOriginShape().GetDim(kHDimNCHWIdx);
        convParas.kw = filterDesc->GetOriginShape().GetDim(kWDimNCHWIdx);
    } else {
        convParas.kh = filterDesc->GetOriginShape().GetDim(kHDimNHWCIdx);
        convParas.kw = filterDesc->GetOriginShape().GetDim(kWDimNHWCIdx);
    }
    convParas.n = cout1Opt * cout0;
    convParas.wci = c1Opt * c0Val;
    convParas.ho = outputDesc->GetShape().GetDim(kHDimNC1HWC0Idx);
    convParas.wo = outputDesc->GetShape().GetDim(kWDimNC1HWC0Idx);
    // get conv attrs
    OP_LOGE_IF(!ge::AttrUtils::GetListInt(opDesc, "pads", padList), false, opType,
        "get attr pads desc failed!");
    OP_LOGE_IF(!ge::AttrUtils::GetListInt(opDesc, "dilations", dilationList), false, opType,
        "get attr dilations desc failed!");
    OP_LOGE_IF(!ge::AttrUtils::GetListInt(opDesc, "strides", stridesList), false, opType,
        "get attr strides desc failed!");
    convParas.padu = padList[kPadUpDimIdx];
    convParas.padd = padList[kPadDownDimIdx];
    convParas.padl = padList[kPadLeftDimIdx];
    convParas.padr = padList[kPadRightDimIdx];
    if (inputDesc->GetOriginFormat() == ge::Format::FORMAT_NCHW) {
        convParas.dilations_h = dilationList[kDilatHDimNCHWIdx];
        convParas.dilations_w = dilationList[kDilatWDimNCHWIdx];
        convParas.stride_h = stridesList[kStriHDimNCHWIdx];
        convParas.stride_w = stridesList[kStriWDimNCHWIdx];
    } else {
        convParas.dilations_h = dilationList[kDilatHDimNHWCIdx];
        convParas.dilations_w = dilationList[kDilatWDimNHWCIdx];
        convParas.stride_h = stridesList[kStriHDimNHWCIdx];
        convParas.stride_w = stridesList[kStriWDimNHWCIdx];
    }

    InitConvUbUtilize(opInfo);

    if (biasDesc != nullptr){
        convParas.biasFlag = true;
    } else {
        convParas.biasFlag = false;
    }

    // data type
    convParas.aType = inputDesc->GetDataType();
    convParas.bType = filterDesc->GetDataType();
    convParas.cType = outputDesc->GetDataType();
    OP_LOGE_IF(CUBE_MAD_TYPE.find(convParas.aType) == CUBE_MAD_TYPE.end(), false, opType, \
        "input datatype only support FP16/INT8/INT4/BFP16/FP32!");
    convParas.madType = CUBE_MAD_TYPE[convParas.aType];
    convParas.biasType = CUBE_MAD_TYPE[convParas.aType];

    GELOGD("[%s] ParserConv2DParas success, input shape is [%d, %d, %d, %d], filter shape is [%d, %d, %d, %d], \
        output shape is [%d, %d, %d, %d], pads is [%d, %d, %d, %d], strides is [%d, %d], dilations is [%d, %d], \
        groups = %d, preFusionUbUtilize = %.3f, preFusionVectorUtilize = %d, postFusionUbUtilize = %.3f, \
        postFusionVectorUtilize = %d, bias_flag = %d, fmap dtype: %d, filter dtype: %d, output dtype: %d, \
        mad type: %d, bias type: %d", nodeName.c_str(), convParas.batch, convParas.fmci, convParas.hi, convParas.wi, \
        convParas.n, convParas.wci, convParas.kh, convParas.kw, convParas.batch, convParas.n, \
        convParas.ho, convParas.wo, convParas.padu, convParas.padd, convParas.padl, convParas.padr, \
        convParas.stride_h, convParas.stride_w, convParas.dilations_h, convParas.dilations_w, convParas.groups, \
        convParas.preFusionUbUtilize, convParas.preFusionVectorUtilize, convParas.postFusionUbUtilize, \
        convParas.postFusionVectorUtilize, convParas.biasFlag, convParas.aType, convParas.bType, \
        convParas.cType, convParas.madType, convParas.biasType);

    return true;
}

/*
 * @brief: check shape and attr range, check exbound l1_size
 */
bool Conv2dBinaryTiling::CheckConv2DParas()
{
    /*
    | Name          | Field   | Scope        |
    | :------------:| :------:| :-----------:|
    | Input size    | H       | [1, 100000]  |
    |               | W       | [1, 4096]    |
    | Filter size   | H       | [1, 255]     |
    |               | W       | [1, 255]     |
    | Stride        | H       | [1, 63]      |
    |               | W       | [1, 63]      |
    | Padding       | Top     | [0, 255]     |
    |               | Bottom  | [0, 255]     |
    |               | Left    | [0, 255]     |
    |               | Right   | [0, 255]     |
    | Dilation      | H       | [1, 255]     |
    |               | W       | [1, 255]     |
    */
    OP_LOGE_IF(!CheckRange(convParas.hi, FMAP_H_LEN_MIN, FMAP_H_LEN_MAX), false, opType, \
        "fmap H only support range [%d, %d]", FMAP_H_LEN_MIN, FMAP_H_LEN_MAX);
    OP_LOGE_IF(!CheckRange(convParas.wi, FMAP_W_LEN_MIN, FMAP_W_LEN_MAX), false, opType, \
        "fmap W only support range [%d, %d]", FMAP_W_LEN_MIN, FMAP_W_LEN_MAX);
    OP_LOGE_IF(!CheckRange(convParas.kh, KERNEL_H_MIN, KERNEL_H_MAX), false, opType, \
        "kernel h only support range [%d, %d]", KERNEL_H_MIN, KERNEL_H_MAX);
    OP_LOGE_IF(!CheckRange(convParas.kw, KERNEL_H_MIN, KERNEL_H_MAX), false, opType, \
        "kernel w only support range [%d, %d]", KERNEL_H_MIN, KERNEL_H_MAX);
    OP_LOGE_IF(!CheckRange(convParas.ho, FMAP_H_LEN_MIN, FMAP_H_LEN_MAX), false, opType, \
        "output H only support range [%d, %d]", FMAP_H_LEN_MIN, FMAP_H_LEN_MAX);
    OP_LOGE_IF(!CheckRange(convParas.wo, FMAP_W_LEN_MIN, FMAP_W_LEN_MAX), false, opType, \
        "output W only support range [%d, %d]", FMAP_W_LEN_MIN, FMAP_W_LEN_MAX);
    OP_LOGE_IF(!CheckRange(convParas.padu, PAD_MIN, PAD_MAX) || !CheckRange(convParas.padu, PAD_MIN, PAD_MAX) || \
        !CheckRange(convParas.padu, PAD_MIN, PAD_MAX) || !CheckRange(convParas.padu, PAD_MIN, PAD_MAX),
        false, opType, "output W only support range [%d, %d]", PAD_MIN, PAD_MAX);
    OP_LOGE_IF(!CheckRange(convParas.dilations_h, DILATION_MIN, DILATION_MAX) || \
        !CheckRange(convParas.dilations_w, DILATION_MIN, DILATION_MAX),
        false, opType, "dilations only support range [%d, %d]", DILATION_MIN, DILATION_MAX);
    OP_LOGE_IF(!CheckRange(convParas.stride_h, STRIDE_MIN, STRIDE_MAX) || \
        !CheckRange(convParas.stride_w, STRIDE_MIN, STRIDE_MAX),
        false, opType, "strides only support range [%d, %d]", STRIDE_MIN, STRIDE_MAX);

    // check in_shape and out_shape
    OP_LOGE_IF(convParas.ho < 1 || convParas.wo < 1, false, opType, \
        "output shape should greater than 0, please check output shape");
    // ho = (fmap_h + padu + padd - (kh - 1) * dilation_h - 1)//stride_h + 1
    int32_t expectHo = (convParas.hi + convParas.padu + convParas.padd - convParas.dilations_h * \
                         (convParas.kh - 1) - 1) / convParas.stride_h + 1;
    int32_t expectWo = (convParas.wi + convParas.padl + convParas.padr - convParas.dilations_w * \
                         (convParas.kw - 1) - 1) / convParas.stride_w + 1;
    OP_LOGE_IF(expectHo != convParas.ho || expectWo != convParas.wo, false, opType, "input and output shape no match!");

    // check L1 size
    OP_LOGE_IF(CheckL1SizeBound(), false, opType, "input range is too large, the mininum tiling may exceed l1_buffer!");
    GELOGD("[%s] CheckConv2DParas success.", nodeName.c_str());

    return true;
}

uint32_t Conv2dBinaryTiling::AlignMN(const uint32_t valueT) {
    return (valueT + GetMKN(convParas.aType, 1) - 1) / GetMKN(convParas.aType, 1);
}

/*
 * @brief: produce attach map
 */
bool Conv2dBinaryTiling::GenAttachMap()
{
    uint32_t fmapReduceK = convParas.fmci * ((convParas.kh - 1) * convParas.dilations_h + 1) * \
                            ((convParas.kw - 1)*convParas.dilations_w + 1);
    uint32_t filterReduceK = convParas.wci * convParas.kh * convParas.kw;

    if (fastTiling.kAl1 == TENSOR_FULL_LOAD && fastTiling.mAl1 == TENSOR_FULL_LOAD) {
        attachMap.al1AttachMode = ATTACH_FULL_LOAD;
    } else if(fastTiling.kAl1 == fmapReduceK) {
        attachMap.al1AttachMode = ATTACH_AT_RES;
    } else {
        attachMap.al1AttachMode = ATTACH_AT_CL0;
    }

    if (fastTiling.kBl1 == TENSOR_FULL_LOAD && fastTiling.nBl1 == TENSOR_FULL_LOAD) {
        attachMap.bl1AttachMode = ATTACH_FULL_LOAD;
        attachMap.bl0AttachMode = ATTACH_NO_FULL_LOAD;
    } else if (fastTiling.kBl1 == FILTER_NO_PASS_L1 && fastTiling.nBl1 == 0) {
        attachMap.bl1AttachMode = FILTER_L1_BYPASS;
        attachMap.bl0AttachMode = (fastTiling.kb == TENSOR_FULL_LOAD) ? ATTACH_FULL_LOAD : ATTACH_NO_FULL_LOAD;
    } else if (fastTiling.kBl1 == filterReduceK) {
        attachMap.bl1AttachMode = ATTACH_AT_RES;
        attachMap.bl0AttachMode = ATTACH_NO_FULL_LOAD;
    } else {
        attachMap.bl1AttachMode = ATTACH_AT_CL0;
        attachMap.bl0AttachMode = ATTACH_NO_FULL_LOAD;
    }

    // bias load mode, default full load
    attachMap.cubChannelwiseMode = 0;

    // N axis only support whole cut and K aixs not whole cut, no need config split_mode
    attachMap.batchSplitMode = (convParas.batch % fastTiling.batchDim == 0) ? INTEGER_SEGMENT : NO_INTEGER_SEGMENT;
    attachMap.groupSplitMode = (convParas.groups % fastTiling.groupDim == 0) ? INTEGER_SEGMENT : NO_INTEGER_SEGMENT;
    GELOGD("[%s] Get Conv template success, al1AttachMode: %d, bl1AttachMode: %d, bl0AttachMode: %d, \
        cubChannelwiseMode: %d, batchSplitMode: %d, groupSplitMode: %d", nodeName.c_str(), attachMap.al1AttachMode,
        attachMap.bl1AttachMode, attachMap.bl0AttachMode, attachMap.cubChannelwiseMode,
        attachMap.batchSplitMode, attachMap.groupSplitMode);

    return true;
}

/*
 * @brief: get tiling and produce attach_mao
 */
bool Conv2dBinaryTiling::GenConv2DTiling(const optiling::Conv2DTilingParseInfo& opInfo)
{
    OP_LOGE_IF(!InitHardwareInfo(opInfo), false, opType, "Get hardwareinfo failed!");
    OP_LOGE_IF(!Conv2dFastTiling(convParas, hardwareInfo, fastTiling), false, opType, "get fasttiling failed!");
    GELOGD("[%s] Conv2dFastTiling success, fastTiling info is: AL0_matrix: [%d, %d, 16, %d], \
        CL0_matrix: [%d, %d, 16, 16, 1, 1], CUB_matrix: [%d, %d, 16, 16], BL0_matrix: [%d, %d, 16, %d, 1, 1], \
        AL1_shape: [%d, %d, 1, 1], AUB_shape: [%d, %d, 1, 1], BL1_shape: [%d, %d, 1, 1], \
        block_dim: [%d, %d, %d, %d]", nodeName.c_str(), fastTiling.ma, fastTiling.ka, \
        GetMKN(convParas.aType, 1), fastTiling.nc, fastTiling.mc, fastTiling.ncFactor, fastTiling.mcFactor, \
        fastTiling.kb, fastTiling.nb, GetMKN(convParas.bType, 1), \
        fastTiling.kAl1, fastTiling.mAl1, fastTiling.kAub, fastTiling.mAub, fastTiling.kBl1, fastTiling.nBl1, \
        fastTiling.batchDim, fastTiling.nDim, fastTiling.mDim, fastTiling.groupDim);

    OP_LOGE_IF(fastTiling.ma == 0, false, opType, "don't support tiling ma = 0.");
    OP_LOGE_IF(fastTiling.ka == 0, false, opType, "don't support tiling ka = 0.");
    OP_LOGE_IF(fastTiling.nb == 0, false, opType, "don't support tiling nb = 0.");
    OP_LOGE_IF(fastTiling.kb == 0, false, opType, "don't support tiling kb = 0.");
    OP_LOGE_IF(fastTiling.ncFactor == 0, false, opType, "don't support tiling ncFactor = 0.");

    convTiling.batchSingleCore = 0; // reserved
    convTiling.nSingleCore = 0; // reserved
    convTiling.batchDim = fastTiling.batchDim;
    convTiling.nDim = fastTiling.nDim;
    convTiling.mDim = fastTiling.mDim;
    convTiling.groupDim = fastTiling.groupDim;
    convTiling.cubN = fastTiling.ncFactor;
    convTiling.nUbL0cFactor = fastTiling.nc / fastTiling.ncFactor;
    convTiling.mL0 = fastTiling.ma;
    convTiling.kL0 = fastTiling.ka;

    if (fastTiling.mAl1 == TENSOR_FULL_LOAD && fastTiling.kAl1 == TENSOR_FULL_LOAD) {
        convTiling.mAl1Factor = AlignMN(convParas.ho * convParas.wo) / (fastTiling.ma * MKN_VALUE_DEFAULT);
        convTiling.kAl116 = convParas.fmci * convParas.kh * convParas.kw / (fastTiling.ka * GetMKN(convParas.bType, 1));
    } else {
        convTiling.mAl1Factor = fastTiling.mAl1;
        convTiling.kAl116 = fastTiling.kAl1 / GetMKN(convParas.bType, 1);
    }

    if (fastTiling.nBl1 == TENSOR_FULL_LOAD && fastTiling.kBl1 == TENSOR_FULL_LOAD) {
        convTiling.nBl1Factor = convParas.n / (fastTiling.nb * MKN_VALUE_DEFAULT);
        convTiling.kBl116 = convParas.wci * convParas.kh * convParas.kw / (fastTiling.kb * GetMKN(convParas.bType, 1));
    } else if (fastTiling.nBl1 == 0 && fastTiling.kBl1 == 0) {
        convTiling.nBl1Factor = 1;
        convTiling.kBl116 = 1;
    } else {
        convTiling.nBl1Factor = fastTiling.nBl1;
        convTiling.kBl116 = fastTiling.kBl1 / GetMKN(convParas.bType, 1);
    }

    OP_LOGE_IF(convTiling.kAl116 == 0, false, opType, "don't support convTiling kAl116 = 0.");
    OP_LOGE_IF(convTiling.kBl116 == 0, false, opType, "don't support convTiling kBl116 = 0.");
    uint32_t fmapReduceK = convParas.fmci * ((convParas.kh - 1) * convParas.dilations_h + 1) * \
                            ((convParas.kw - 1)*convParas.dilations_w + 1);
    uint32_t filterReduceK = convParas.wci * convParas.kh * convParas.kw;
    convTiling.kAl1Factor = fmapReduceK / convTiling.kAl116 / GetMKN(convParas.bType, 1);
    convTiling.kBl1Factor = filterReduceK / convTiling.kBl116 / GetMKN(convParas.bType, 1);
    // need add kub aub
    convTiling.kAub = fastTiling.kAub;
    convTiling.mAub = fastTiling.mAub;
    GELOGD("[%s] Get convTiling from fastTiling success, batchSingleCore = %d, nSingleCore = %d, batchDim = %d, \
        nDim = %d, mDim = %d, groupDim = %d, cubN = %d, nUbL0cFactor = %d, mL0 = %d, kL0 = %d, mAl1Factor = %d, \
        nBl1Factor = %d, kAl116 = %d, kBl116 = %d, kAl1Factor = %d, kBl1Factor = %d, kAub = %d, mAub = %d", \
        nodeName.c_str(), convTiling.batchSingleCore, convTiling.nSingleCore, convTiling.batchDim, \
        convTiling.nDim, convTiling.mDim, convTiling.groupDim, convTiling.cubN, convTiling.nUbL0cFactor, \
        convTiling.mL0, convTiling.kL0, convTiling.mAl1Factor, convTiling.nBl1Factor, convTiling.kAl116, \
        convTiling.kBl116, convTiling.kAl1Factor, convTiling.kBl1Factor, convTiling.kAub, convTiling.mAub);

    OP_LOGE_IF(!GenAttachMap(), false, opType, "GenAttachMap failed!");

    return true;
}

/*
 * @brief: update conv2d runinfo
 */
bool Conv2dBinaryTiling::SetRunInfo()
{
    runParas.dilationH = convParas.dilations_h;
    runParas.dilationW = convParas.dilations_w;
    runParas.strideH = convParas.stride_h;
    runParas.strideW = convParas.stride_w;
    runParas.batch = convParas.batch;
    runParas.hi = convParas.hi;
    runParas.ho = convParas.ho;
    runParas.wi = convParas.wi;
    runParas.wo = convParas.wo;
    runParas.c1In = convParas.fmci / GetMKN(convParas.aType, 1);
    runParas.c1Out = convParas.n / GetMKN(convParas.bType, MKN_NINDEX);
    runParas.kh = convParas.kh;
    runParas.kw = convParas.kw;
    runParas.padu = convParas.padu;
    runParas.padd = convParas.padd;
    runParas.padl = convParas.padl;
    runParas.padr = convParas.padr;
    runParas.batchSingleCore = convTiling.batchSingleCore;
    runParas.nSingleCore = convTiling.nSingleCore;
    runParas.batchDim = convTiling.batchDim;
    runParas.nDim = convTiling.nDim;
    runParas.mDim = convTiling.mDim;
    runParas.groupDim = convTiling.groupDim;
    runParas.cubN = convTiling.cubN;
    runParas.nUbL0cFactor = convTiling.nUbL0cFactor;
    runParas.mL0 = convTiling.mL0;
    runParas.kL0 = convTiling.kL0;
    runParas.mAl1Factor = convTiling.mAl1Factor;
    runParas.nBl1Factor = convTiling.nBl1Factor;
    runParas.kAl116 = convTiling.kAl116;
    runParas.kBl116 = convTiling.kBl116;
    runParas.kAl1Factor = convTiling.kAl1Factor;
    runParas.kBl1Factor = convTiling.kBl1Factor;
    GELOGD("[%s] AddTilingData success, tilingdata is %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, \
        %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d", nodeName.c_str(), runParas.dilationH, \
        runParas.dilationW, runParas.strideH, runParas.strideW, runParas.batch, runParas.hi, \
        runParas.ho, runParas.wi, runParas.wo, runParas.c1In, runParas.c1Out, runParas.kh, runParas.kw, \
        runParas.padu, runParas.padd, runParas.padl, runParas.padr, runParas.batchSingleCore, \
        runParas.nSingleCore, runParas.batchDim, runParas.nDim, runParas.mDim, runParas.cubN, \
        runParas.nUbL0cFactor, runParas.mL0, runParas.kL0, runParas.mAl1Factor, \
        runParas.nBl1Factor, runParas.kAl116, runParas.kBl116, runParas.kAl1Factor, \
        runParas.kBl1Factor);

    return true;
}

/*
 * @brief: update kernel_id and tilingdata
 * @param [out] runInfo
 */
bool Conv2dBinaryTiling::UpdateRunInfo(utils::OpRunInfo& runInfo)
{
    bool load2dFlag = (convParas.padu + convParas.padd + convParas.padl + convParas.padr) == 0 &&
        convParas.stride_h*convParas.stride_w == 1 && convParas.kh*convParas.kw == 1 &&
        convParas.bType == ge::DataType::DT_FLOAT16;
    attachMap.fmapLoadtol0aMode = load2dFlag ? 0 : 1;

    /*
        tilingId is int64 type, the meaning of each bits as follows:
        0 ~ 2bit : al1AttachMode
        3 ~ 5bit : bl1AttachMode
        6 ~ 8bit : bl0AttachMode
        9 ~ 10bit : batchSplitMode
        11 ~ 12bit: groupSplitMode
        13 ~ 14bit: cubChannelwiseMode
        15 ~ 16bit: fmapLoadtol0aMode
    */
    uint64_t tilingId = 0;
    tilingId = tilingId | (attachMap.al1AttachMode << 0);
    tilingId = tilingId | (attachMap.bl1AttachMode << BIT_BL1_LOC);
    tilingId = tilingId | (attachMap.bl0AttachMode << BIT_BL0_LOC);
    tilingId = tilingId | (attachMap.batchSplitMode << BIT_BATCH_SPLIT_LOC);
    tilingId = tilingId | (attachMap.groupSplitMode << BIT_GROUP_SPLIT_LOC);
    tilingId = tilingId | (attachMap.cubChannelwiseMode << BIT_CHANNELWISE_LOC);
    tilingId = tilingId | (attachMap.fmapLoadtol0aMode << BIT_LOADMODE_LOC);
    bitset<ATTACH_BITS_LEN> bitValue(tilingId);
    string bitStr = bitValue.to_string();
    GELOGD("[%s] Get tilingId bitmap string format : %s", nodeName.c_str(), bitStr.c_str());

    runInfo.SetTilingKey(static_cast<uint64_t>(tilingId));
    GELOGD("[%s] SetTilingKey success, tilingId is %d", nodeName.c_str(), tilingId);

    int32_t blockNum = convTiling.batchDim * convTiling.nDim * convTiling.mDim * convTiling.groupDim;
    runInfo.SetBlockDim(static_cast<uint32_t>(blockNum));
    GELOGD("[%s] SetBlockDim success, blockNum is %d", nodeName.c_str(), blockNum);

    OP_LOGE_IF(!SetRunInfo(), false, opType, "SetRunInfo failed!");
    runInfo.AddTilingData(runParas);

    return true;
}

bool ProduceBinaryTiling(const ge::OpDescPtr opDesc, optiling::Conv2DTilingParseInfo& opInfo,
                         utils::OpRunInfo& runInfo)
{
    string opType = opDesc->GetType();
    unique_ptr <Conv2dBinaryTiling> binaryTilingPtr(new Conv2dBinaryTiling());
    OP_LOGE_IF(!binaryTilingPtr->ParserConv2DParas(opDesc, opInfo), false, opType, "parser conv2d params failed!");
    OP_LOGE_IF(!binaryTilingPtr->CheckConv2DParas(), false, opType, "check conv2d params failed!");
    OP_LOGE_IF(!binaryTilingPtr->GenConv2DTiling(opInfo), false, opType, "GenConv2DTiling failed!");

    return binaryTilingPtr->UpdateRunInfo(runInfo);
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
    OP_LOGE_IF(inputDesc == nullptr, false, opType, "GetInputDescPtr failed!");

    auto outputDesc = opDesc->GetOutputDescPtr(0);
    OP_LOGE_IF(outputDesc == nullptr, false, opType, "GetOutputDescPtr failed!");

    ge::Format inputFormat = inputDesc->GetFormat();
    std::string xFormat = ge::TypeUtils::FormatToSerialString(inputFormat).c_str();
    if (xFormat != "NC1HWC0" && xFormat != "NHWC") {
        OP_LOGE(opType.c_str(), "only support NC1HWC0 or NHWC format.");
    }

    if (opInfo.tilingType == "binary_tiling") {
        GELOGD("[%s] optiling type is binary_tiling", opDesc->GetName().c_str());
        return ProduceBinaryTiling(opDesc, opInfo, runInfo);
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
