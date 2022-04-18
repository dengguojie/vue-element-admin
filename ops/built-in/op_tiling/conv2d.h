/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file conv2d.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CONV2D_DSL_H_
#define OPS_BUILT_IN_OP_TILING_CONV2D_DSL_H_

#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "conv2d_fast_tiling.h"
#include <nlohmann/json.hpp>
#include "cube_tiling.h"
#include "graph/debug/ge_log.h"
#include "external/graph/operator.h"
#include "op_tiling_util.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling
{
const int32_t kNCHWDimSize = 4;
const int32_t kNC1HWC0DimSize = 5;
const int32_t kFRACZDimSize = 4;

// FRACZ shape (C1*kh*kw, N1, N0, C0)
const int32_t kC1DimFRACZIdx = 0;
const int32_t kC0DimFRACZIdx = 3;

// NC1HWC0 shape
const int32_t kNDimNC1HWC0Idx = 0;
const int32_t kC1DimNC1HWC0Idx = 1;
const int32_t kHDimNC1HWC0Idx = 2;
const int32_t kWDimNC1HWC0Idx = 3;
const int32_t kC0DimNC1HWC0Idx = 4;

// NCHW shape
const int32_t kNDimNCHWIdx = 0;
const int32_t kCDimNCHWIdx = 1;
const int32_t kHDimNCHWIdx = 2;
const int32_t kWDimNCHWIdx = 3;

// NHWC shape
const int32_t kNDimNHWCIdx = 0;
const int32_t kCDimNHWCIdx = 3;
const int32_t kHDimNHWCIdx = 1;
const int32_t kWDimNHWCIdx = 2;

// PAD IDX
const int32_t kPadUpDimIdx = 0;
const int32_t kPadDownDimIdx = 1;
const int32_t kPadLeftDimIdx = 2;
const int32_t kPadRightDimIdx = 3;

// STRIDES IDX
const int32_t kStriHDimNCHWIdx = 2;
const int32_t kStriWDimNCHWIdx = 3;

const int32_t kStriHDimNHWCIdx = 1;
const int32_t kStriWDimNHWCIdx = 2;

// DILATION IDX
const int32_t kDilatHDimNCHWIdx = 2;
const int32_t kDilatWDimNCHWIdx = 3;

const int32_t kDilatHDimNHWCIdx = 1;
const int32_t kDilatWDimNHWCIdx = 2;

// L1 BUFFER SIZE: 1M
const int32_t L1_BUFFER_SIZE = 1024 * 1024;

// full load flag
const uint32_t TENSOR_FULL_LOAD = UINT32_MAX;
const uint32_t FILTER_NO_PASS_L1 = 0;

// attach mode
const int32_t ATTACH_FULL_LOAD = 0;
const int32_t ATTACH_NO_FULL_LOAD = 1;
const int32_t ATTACH_AT_RES = 1;
const int32_t ATTACH_AT_CL0 = 2;
const int32_t FILTER_L1_BYPASS = 3;
const int32_t INTEGER_SEGMENT = 0;
const int32_t NO_INTEGER_SEGMENT = 1;

// shape and attr range
const int32_t FMAP_H_LEN_MIN = 1;
const int32_t FMAP_H_LEN_MAX = 100000;
const int32_t FMAP_W_LEN_MIN = 1;
const int32_t FMAP_W_LEN_MAX = 4096;
const int32_t KERNEL_H_MIN = 1;
const int32_t KERNEL_H_MAX = 255;
const int32_t PAD_MIN = 0;
const int32_t PAD_MAX = 255;
const int32_t STRIDE_MIN = 1;
const int32_t STRIDE_MAX = 63;
const int32_t DILATION_MIN = 1;
const int32_t DILATION_MAX = 255;

const int32_t kModeCoeficient = 10;

const int32_t ATTACH_BITS_LEN = 64;
const int32_t BIT_BL1_LOC = 3;
const int32_t BIT_BL0_LOC = 6;
const int32_t BIT_BATCH_SPLIT_LOC = 9;
const int32_t BIT_GROUP_SPLIT_LOC = 11;
const int32_t BIT_CHANNELWISE_LOC = 13;
const int32_t BIT_LOADMODE_LOC = 15;

const int32_t MKN_VALUE_DEFAULT = 16;
const int32_t MKN_NINDEX = 2;
const std::vector<int32_t> INT4_MKN_MAP = {16, 64, 16};
const std::vector<int32_t> INT8_MKN_MAP = {16, 32, 16};
const std::vector<int32_t> FP16_MKN_MAP = {16, 16, 16};
const std::vector<int32_t> FLOAT_MKN_MAP = {16, 8, 16};
std::map<ge::DataType, std::vector<std::int32_t>> CUBE_MKN_MAP = {
    {ge::DataType::DT_INT4, INT4_MKN_MAP},
    {ge::DataType::DT_INT8, INT8_MKN_MAP},
    {ge::DataType::DT_UINT8, INT8_MKN_MAP},
    {ge::DataType::DT_INT16, FP16_MKN_MAP},
    {ge::DataType::DT_INT32, FP16_MKN_MAP},
    {ge::DataType::DT_FLOAT16, FP16_MKN_MAP},
    {ge::DataType::DT_FLOAT, FLOAT_MKN_MAP},
    {ge::DataType::DT_BF16, FP16_MKN_MAP}
};

std::map<ge::DataType, float> M_BIT_RATIO = {
    {ge::DataType::DT_INT4, 0.5},
    {ge::DataType::DT_INT8, 1.0},
    {ge::DataType::DT_UINT8, 1.0},
    {ge::DataType::DT_FLOAT16, 2.0},
    {ge::DataType::DT_INT16, 2.0},
    {ge::DataType::DT_BF16, 2.0},
    {ge::DataType::DT_FLOAT, 4.0}
};

std::map<ge::DataType, ge::DataType> CUBE_MAD_TYPE = {
    {ge::DataType::DT_INT4, ge::DataType::DT_INT32},
    {ge::DataType::DT_INT8, ge::DataType::DT_INT32},
    {ge::DataType::DT_UINT8, ge::DataType::DT_INT32},
    {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_BF16, ge::DataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT}
};

struct Conv2DRunInfo
{
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t batch = 1;
    uint32_t hi = 1;
    uint32_t ho = 1;
    uint32_t wi = 1;
    uint32_t wo = 1;
    uint32_t c1In = 1;
    uint32_t c1Out = 1;
    uint32_t kh = 1;
    uint32_t kw = 1;
    uint32_t padu = 0;
    uint32_t padd = 0;
    uint32_t padl = 0;
    uint32_t padr = 0;
    uint32_t batchSingleCore = 0;
    uint32_t nSingleCore = 0;
    uint32_t batchDim = 0;
    uint32_t nDim = 0;
    uint32_t mDim = 0;
    uint32_t groupDim = 0;
    uint32_t cubN = 0;
    uint32_t nUbL0cFactor = 0;
    uint32_t mL0 = 0;
    uint32_t kL0 = 0;
    uint32_t mAl1Factor = 0;
    uint32_t nBl1Factor = 0;
    uint32_t kAl116 = 0;
    uint32_t kBl116 = 0;
    uint32_t kAl1Factor = 0;
    uint32_t kBl1Factor = 0;
};

struct Conv2DCacheTiling
{
    uint32_t batchSingleCore = 0;
    uint32_t nSingleCore = 0;
    uint32_t batchDim = 0;
    uint32_t nDim = 0;
    uint32_t mDim = 0;
    uint32_t groupDim = 0;
    uint32_t cubN = 0;
    uint32_t nUbL0cFactor = 0;
    uint32_t mL0 = 0;
    uint32_t kL0 = 0;
    uint32_t mAl1Factor = 0;
    uint32_t nBl1Factor = 0;
    uint32_t kAl116 = 0;
    uint32_t kBl116 = 0;
    uint32_t kAl1Factor = 0;
    uint32_t kBl1Factor = 0;
    uint32_t kAub = 0;
    uint32_t mAub = 0;
};

struct Conv2DAttachMap
{
    uint32_t al1AttachMode = 0;
    uint32_t bl1AttachMode = 0;
    uint32_t bl0AttachMode = 0;
    uint32_t batchSplitMode = 0;
    uint32_t groupSplitMode = 0;
    uint32_t cubChannelwiseMode = 0;
    uint32_t fmapLoadtol0aMode = 0;  // 0: load2d, 1:load3d
};

uint32_t Lcm(const uint32_t valueA, const uint32_t valueB);

class Conv2dBinaryTiling {
public:
    explicit Conv2dBinaryTiling() {};
    virtual ~Conv2dBinaryTiling() {};

    bool ParserConv2DParas(const ge::OpDescPtr& op_desc, const optiling::Conv2DTilingParseInfo& opInfo);
    bool CheckConv2DParas();
    bool GenConv2DTiling();
    bool UpdateRunInfo(utils::OpRunInfo& runInfo);

private:
    uint32_t GetMKN(ge::DataType dType, uint32_t idx);
    inline bool CheckRange(int32_t value, int32_t lowerLimit, int32_t upperLimit);
    bool InitConvUbUtilize();
    bool InitHardwareInfo();
    bool CheckL1SizeBound();
    uint32_t AlignMN(const uint32_t valueT);
    bool GenAttachMap();
    bool SetRunInfo();

    optiling::Conv2dParams convParas;
    optiling::HardwareInfo hardwareInfo;
    optiling::Conv2dTiling fastTiling;
    Conv2DCacheTiling convTiling;
    Conv2DAttachMap attachMap;
    Conv2DRunInfo runParas;

    std::string opType;
    std::string nodeName;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_CONV2D_DSL_H_