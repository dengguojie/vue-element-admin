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
 * \file conv2d_fast_tiling.h
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"

#define CHECK_OP_FUNC(cond, post_action_expr, msg, ...)                                                          \
  {                                                                                                              \
    if (cond) {                                                                                                  \
      CUBE_INNER_ERR_REPORT("Conv2d", msg, ##__VA_ARGS__);                                                       \
      post_action_expr;                                                                                          \
    }                                                                                                            \
  }


#define CHECK_FAST_TILING_DATA_RANGE(data, min, max, name, ...)                                                  \
{                                                                                                                \
    bool isValueInRange = (data >= min && data <= max);                                                          \
    CHECK_OP_FUNC(                                                                                               \
        !isValueInRange, return false, "Invalid %s, except value in [%ld, %ld], but got %ld",                    \
        name, static_cast<long>(min), static_cast<long>(max), static_cast<long>(data));                          \
}                                                                                                                \

namespace optiling
{
enum class PBuffer {
    PBUFFER_OFF = 1,
    PBUFFER_DB = 2,
    PBUFFER_QUAD = 4,
};

enum class FastTilingValue : uint32_t {
    FAST_TILING_VALUE_1 = 1,
    FAST_TILING_VALUE_2 = 2,
    FAST_TILING_VALUE_3 = 3,
    FAST_TILING_VALUE_4 = 4,
    FAST_TILING_VALUE_128 = 128,
};

const uint32_t CUBE_UNIT_32 = 32;
const uint32_t CUBE_UNIT_16 = 16;
const uint32_t FULL_LOAD = UINT32_MAX;
const uint32_t FOUR = 4;
const uint32_t THREE = 3;
const uint32_t TWO = 2;
const float HALF = 0.5;
const uint32_t MSIZE_PER_CORE = 4096;
const int64_t MAX_WIDTH_SIZE = 4096;
const int64_t MIN_WIDTH_SIZE = 1;
const int64_t MAX_HEIGHT_SIZE = 100000;
const int64_t MIN_HEIGHT_SIZE = 1;
const int64_t MAX_PADDING_SIZE = 255;
const int64_t MAX_DILATION_SIZE = 255;
const int64_t MAX_STRIDE_SIZE = 63;

struct TilingRangeL1
{
  std::vector<uint32_t> mAL1 = {0};
  std::vector<uint32_t> kAL1 = {0};
  std::vector<uint32_t> group = {0};
  std::vector<uint32_t> batch = {0};
  std::vector<uint32_t> nBL1 = {0};
  std::vector<uint32_t> kBL1 = {0};
  uint32_t pbuffer = static_cast<uint32_t>(PBuffer::PBUFFER_DB);
  std::vector<uint32_t> pbAL1 = {pbuffer};
  std::vector<uint32_t> pbBL1 = {pbuffer};
};

struct BlockDimRange
{
    std::vector<uint32_t> batchRange;
    std::vector<uint32_t> nRange;
    std::vector<uint32_t> mRange;
    std::vector<uint32_t> gRange;
};

struct HardwareInfo
{
    uint32_t aicoreNum = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    uint64_t btSize = 0;
    uint32_t ddrReadRate = 0;
    uint32_t ddrWriteRate = 0;
    uint32_t l2Rate = 0;
    uint32_t l2ReadRate = 0;
    uint32_t l2WriteRate = 0;
    uint32_t l1ToL0aRate = 0;
    uint32_t l1ToL0bRate = 0;
    uint32_t l1ToUbRate = 0;
    uint32_t l0cToUbRate = 0;
    uint32_t ubToL2Rate = 0;
    uint32_t ubToDdrRate = 0;
    uint32_t ubToL1Rate = 0;
    uint32_t cubeBandwidth = 0;
    uint32_t vectorBandwidth = 0;
    bool cubeVectorSplit = false;
    std::string socVersion = "";
};

struct Shape5D
{
    std::vector<int64_t> iShape5D; // 5d input feature-map dims
    std::vector<int64_t> wShape5D; // 5d input weights dims
    std::vector<int64_t> oShape5D; // 5d output dims
};

struct Conv2dParams
{
    int64_t batch = 0;
    int64_t fmci = 0;
    int64_t hi = 0;
    int64_t wi = 0;
    int64_t n = 0;
    int64_t wci = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t ho = 0;
    int64_t wo = 0;
    int64_t padu = 0;
    int64_t padd = 0;
    int64_t padl = 0;
    int64_t padr = 0;
    int64_t dilations_h = 1;
    int64_t dilations_w = 1;
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t groups = 1;
    float preFusionUbUtilize = 0;
    int64_t preFusionVectorUtilize = 0;
    float postFusionUbUtilize = 0;
    int64_t postFusionVectorUtilize = 0;

    bool biasFlag = false;
    ge::DataType aType = ge::DataType::DT_FLOAT16;
    ge::DataType bType = ge::DataType::DT_FLOAT16;
    ge::DataType cType = ge::DataType::DT_FLOAT16;
    ge::DataType madType = ge::DataType::DT_FLOAT;
    ge::DataType biasType = ge::DataType::DT_FLOAT16;
};

struct Conv2dTiling
{
    uint32_t batchDim = 0;
    uint32_t nDim = 0;
    uint32_t mDim = 0;
    uint32_t groupDim = 0;
    uint32_t kAl1 = 0;
    uint32_t mAl1 = 0;
    uint32_t kBl1 = 0;
    uint32_t nBl1 = 0;
    uint32_t ma = 0;
    uint32_t ka = 0;
    uint32_t kb = 0;
    uint32_t nb = 0;
    uint32_t mc = 0;
    uint32_t nc = 0;
    uint32_t ncFactor = 0;
    uint32_t mcFactor = 0;
    uint32_t kAub = 0;
    uint32_t mAub = 0;
};

struct Tiling
{
    // block dim
    uint32_t batchDim = 1;
    uint32_t nDim = 1;
    uint32_t mDim = 1;
    uint32_t groupDim = 1;
    // L1A
    uint32_t kAL1 = 1;
    uint32_t mAL1 = 1;
    uint32_t kAL1ci = 1;
    uint32_t mAL1Value = 1;
    uint32_t batchAL1 = 1;
    uint32_t groupAL1 = 1;
    uint32_t al1DBuffer = 1;
    // L1B
    uint32_t kBL1 = 1;
    uint32_t nBL1 = 1;
    uint32_t kBL1ci = 1;
    uint32_t nBL1Value = 1;
    uint32_t groupBL1 = 1;
    uint32_t bl1DBuffer = 1;
    // L0C
    uint32_t mC = 1;
    uint32_t nC = 1;
    uint32_t batchCL0 = 1;
    uint32_t groupCL0 = 1;
    uint32_t cl0PBuffer = 1;
    // L0A
    uint32_t kA = 1;
    uint32_t mA = 1;
    uint32_t batchAL0 = 1;
    uint32_t groupAL0 = 1;
    uint32_t al0PBuffer = 1;
    // L0B
    uint32_t kB = 1;
    uint32_t nB = 1;
    uint32_t batchBL0 = 1;
    uint32_t groupBL0 = 1;
    uint32_t bl0PBuffer = 1;
    // CUB
    uint32_t nCFactor = 1;
    uint32_t mCFactor = 1;
    uint32_t batchCUB = 1;
    uint32_t groupCUB = 1;
    uint32_t l0cOutputPBuffer = 1;
    uint32_t cubPBuffer = 1;
    // AUB
    uint32_t kAub = 1;
    uint32_t mAub = 1;
    uint32_t batchAUB = 1;
    uint32_t groupAUB = 1;
    uint32_t aubPBuffer = 1;
};

struct BlockDimData
{
   uint32_t batchIndex;
   uint32_t nIndex;
   uint32_t mIndex;
   uint32_t gIndex;
};

struct BlockDimSize
{
   uint32_t iSize;
   uint32_t wSize;
   uint32_t oSize;
};

struct L1Data
{
    uint32_t kAL1_index = 0;
    uint32_t kBL1_index = 0;
    uint32_t nBL1_index = 0;
    uint32_t mAL1_index = 0;
    uint32_t fmCurrent = 0;
    uint32_t filter_current = 0;
    uint32_t result_current = 0;
    uint32_t l1_current = 0;
};

struct TilingRangeL0
{
    std::vector<uint32_t> mL0 = {0};
    std::vector<uint32_t> kL0 = {0};
    std::vector<uint32_t> kAL0 = {0};
    std::vector<uint32_t> kBL0 = {0};
    std::vector<uint32_t> nL0 = {0};
    std::vector<uint32_t> groupL0 = {0};
    std::vector<uint32_t> batchL0 = {0};
    uint32_t pbuffer = static_cast<uint32_t>(PBuffer::PBUFFER_DB);
    std::vector<uint32_t> pbAL0 = {pbuffer};
    std::vector<uint32_t> pbBL0 = {pbuffer};
    std::vector<uint32_t> pbCL0 = {pbuffer};
};

struct L0Data
{
    uint32_t kL0Index = 0;
    uint32_t nL0Index = 0;
    uint32_t mL0Index = 0;
    uint32_t l0ACurrent = 0;
    uint32_t l0BCurrent = 0;
    uint32_t l0CCurrent = 0;
    bool updateFlag = true;
};

struct TilingRangeUB
{
    std::vector<uint32_t> kAub = {0};
    std::vector<uint32_t> mAub = {0};
    std::vector<uint32_t> ncFactor = {0};
    std::vector<uint32_t> mcFactor = {0};
    uint32_t pbuffer = static_cast<uint32_t>(PBuffer::PBUFFER_OFF);
    std::vector<uint32_t> pbAUB = {pbuffer};
    std::vector<uint32_t> pbCUB = {pbuffer};
    std::vector<uint32_t> pbUBG = {pbuffer};
};

struct UBData
{
    uint32_t kAubIndex = 0;
    uint32_t mAubIndex = 0;
    uint32_t ncFactorIndex = 0;
    uint32_t mcFactorIndex = 0;
    uint32_t preUbCurrent = 0;
    uint32_t postUbCurrent = 0;
    bool updateFlag = true;
};

bool Conv2dFastTiling(const Conv2dParams& inputParams, const HardwareInfo& hardwareInfo, Conv2dTiling& tiling);

class FastTiling {
public:
    explicit FastTiling() {};
    virtual ~FastTiling() {};

    bool SetInputParams(const Conv2dParams& inputParams, const HardwareInfo& hardwareInfo);
    bool GetConv2dTiling(Tiling& tiling);
    bool InfoTranslate(const Tiling& tiling, Conv2dTiling& conv2dTiling);
private:
    bool CheckConv2dParams(const Conv2dParams& conv2dInfo);
    bool CheckHardwareInfo(const HardwareInfo& hardwareInfo);
    void GetConv2dCaseStatus();
    void Convert4DTo5D();
    // infer block dim tiling at runtime
    float GetBlockDimCompTime(const Tiling& tiling,
                              const BlockDimSize& dataPerCore,
                              const BlockDimSize& blockDimSize);
    void GetBlockDimRange(BlockDimRange& tilingRange);
    bool GetBlockDimTiling(Tiling& tiling);
    void DataInOneCore(const BlockDimSize& dataSize,
                       const Tiling& blockDim,
                       BlockDimSize& dataPerCore);
    void GetL1TilingRange(Tiling &tiling);
    bool GetL1Tiling(Tiling &tiling);
    void AssignmentL1(Tiling &tiling);
    // get L0 dim range
    void WeightFullLoad();
    void GetL0TilingRange(const Tiling& tiling);
    // infer L0 Tiling
    bool GetL0Tiling(Tiling& tiling);
    void WeightL1ToL0Reset(Tiling &tiling);
    void WeightL1ToL0();
    bool L1ToL0SizeSatify();
    void AssignmentL0(Tiling &tiling);
    // get UB dim range
    void GetUBTilingRange(const Tiling& tiling);
    // infer UB Tiling
    bool GetUBTiling(Tiling& tiling);
    void UpdateUBData();
    void UpdateL0Data();
    void AddmL0();
    void AddkL0();
    void AddnL0();
    void AddmAub();
    void AddkAub();

    void AddL1Data(const pair<string, float>& data);
    void AddNbl1();
    void AddMal1();
    void AddKal1();
    void AddKbl1();

    uint32_t getCi0();
    int64_t GetEleNum(const vector<int64_t>& shape);

    void UpdateL1Data();

    static void CalcCommFactor(const uint32_t num, const uint32_t numMax, std::vector<uint32_t>& rlist);
    // hardware info
    HardwareInfo hardware_;
    // op infos
    Conv2dParams opInfo_;
    Shape5D shapeInfo_;
    TilingRangeL1 tilingRangeL1_; // every dim range in L1
    TilingRangeL0 tilingRangeL0_; // every dim range in L0
    TilingRangeUB tilingRangeUB_; // every dim range in L0
    L1Data l1Data_; // attr of L1 in current case
    L0Data l0Data_; // attr of L0 in current case
    UBData ubData_; // attr of L0 in current case
    uint32_t khDilated_;
    uint32_t kwDilated_;
    uint32_t reduceKAxisAL1_KhDilKwDilCi0_;
    uint32_t reduceKAxisBL1_KhKwCi0_;
    uint32_t isSplitWAxis_;
    // conv2d case status
    bool isMultiGroupL0cFlag = false;
    bool isLoad2dFlag = false;
    bool isFmReadWithStrideFlag = false;
    bool isQuantScene_ = false; // flag for quant scene

    const std::unordered_map<ge::DataType, uint32_t> byteForDtype_ = {
        {ge::DataType::DT_FLOAT, static_cast<uint32_t>(FastTilingValue::FAST_TILING_VALUE_4)},
        {ge::DataType::DT_FLOAT16, static_cast<uint32_t>(FastTilingValue::FAST_TILING_VALUE_2)},
        {ge::DataType::DT_INT32, static_cast<uint32_t>(FastTilingValue::FAST_TILING_VALUE_4)},
    };
};
}