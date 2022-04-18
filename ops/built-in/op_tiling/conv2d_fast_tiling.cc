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
#include <cfloat>

#include "conv2d_fast_tiling.h"

using namespace std;

namespace optiling {
bool Conv2dFastTiling(const Conv2dParams& inputParams, const HardwareInfo& hardwareInfo, Conv2dTiling& tiling)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    Tiling generalTiling;
    CHECK_OP_FUNC(!(fastTilingPtr->GetConv2dTiling(generalTiling)), return false, "compute Conv2dTiling failed");
    // convert general tiling into conv2d operator specific tiling
    return fastTilingPtr->InfoTranslate(generalTiling, tiling);
}

void FastTiling::Convert4DTo5D()
{
    const int64_t batch = opInfo_.batch;
    const int64_t ci1 = (opInfo_.fmci + getCi0() - 1) / getCi0();
    const int64_t hi = opInfo_.hi;
    const int64_t wi = opInfo_.wi;
    const int64_t ci0 = getCi0();
    opInfo_.iShape5D = {batch, ci1, hi, wi, ci0};

    const int64_t cout = opInfo_.n;
    const int64_t kernnelCi1 = (opInfo_.wci + getCi0() - 1) / getCi0();
    const int64_t kernnelHi = opInfo_.kh;
    const int64_t kernnelWi = opInfo_.kw;
    const int64_t kernnelCi0 = getCi0();
    opInfo_.wShape5D = {cout, kernnelCi1, kernnelHi, kernnelWi, kernnelCi0};

    const int64_t co1 = (cout + getCi0() - 1) / getCi0();
    const int64_t ho = opInfo_.ho;
    const int64_t wo = opInfo_.wo;
    const int64_t co0 = getCi0();
    opInfo_.oShape5D = {batch, co1, ho, wo, co0};
}

void FastTiling::SetInputParams(const Conv2dParams& inputParams, const HardwareInfo& hardwareInfo)
{
    this->opInfo_ = inputParams;
    GetConv2dCaseStatus();
    this->Convert4DTo5D(); // set 5d shape
    this->hardware_ = hardwareInfo;
    this->hardware_.cubeBandwidth = CUBE_UNIT_16 * CUBE_UNIT_16 * CUBE_UNIT_16 *
                                    byteForDtype_.at(ge::DataType::DT_FLOAT16);
    uint32_t vectorProcessNum = static_cast<uint32_t>(FastTilingValue::FAST_TILING_VALUE_128);
    this->hardware_.vectorBandwidth = vectorProcessNum * byteForDtype_.at(ge::DataType::DT_FLOAT16);
}

bool FastTiling::GetConv2dTiling(Tiling& tiling) {
    CHECK_OP_FUNC(!GetBlockDimTiling(tiling), return false, "get block dim failed");
    CHECK_OP_FUNC(!GetL1Tiling(tiling), return false, "get l1 tling failed");
    CHECK_OP_FUNC(!GetL0Tiling(tiling), return false, "get l0 tiling failed");
    CHECK_OP_FUNC(!GetUBTiling(tiling), return false, "get ub tiling failed");
    return true;
}

bool FastTiling::InfoTranslate(const Tiling& tiling, Conv2dTiling& conv2dTiling)
{
    conv2dTiling.batchDim = tiling.batchDim;
    conv2dTiling.nDim = tiling.nDim;
    conv2dTiling.mDim = tiling.mDim;
    conv2dTiling.groupDim = tiling.groupDim;
    conv2dTiling.kAl1 = tiling.kAL1;
    conv2dTiling.mAl1 = tiling.mAL1;
    conv2dTiling.kBl1 = tiling.kBL1;
    conv2dTiling.nBl1 = tiling.nBL1;
    conv2dTiling.ma = tiling.mA;
    conv2dTiling.ka = tiling.kA;
    conv2dTiling.kb = tiling.kB;
    conv2dTiling.nb = tiling.nB;
    conv2dTiling.mc = tiling.mC;
    conv2dTiling.nc = tiling.nC;
    conv2dTiling.ncFactor = tiling.nCFactor;
    conv2dTiling.mcFactor = tiling.mCFactor;
    conv2dTiling.kAub = tiling.kAub;
    conv2dTiling.mAub = tiling.mAub;

    return true;
}

/* calculate the data amount distributed to different ai cores
 * @param        dataSize      vector<uint32_t> {iSize, wSize, oSize}
 * @param        blockDim      vector<uint32_t> {batch, n, m, group}
 * @return       dataPerCore   vector<uint32_t> {iSizeL1, wSizeL1, oSizeL1}
 */
void FastTiling::DataInOneCore(const BlockDimSize& dataSize,
                               const Tiling& blockDim,
                               BlockDimSize& dataPerCore)
{
    /*
     *   eg.
     *      m dim = 2; X represents overlapping pixels
     *                   wi                                   kw
     *   |-------|-------|-------|-------|        |-------|-------|-------|        |-------|-------|
     *   |       |       |       |       |        |       |       |       |        |       |       |
     *   |-------|-------|-------|-------|        |-------|-------|-------|     ---|-------|-------|---  m dim =2
     *   |   X   |   X   |   X   |   X   |      kh|       |       |       |        |       |       |
     * hi|-------|-------|-------|-------|        |-------|-------|-------|        |-------|-------|
     *   |   X   |   X   |   X   |   X   |        |       |       |       |               out
     *   |-------|-------|-------|-------|        |-------|-------|-------|
     *   |       |       |       |       |                  filter
     *   |-------|-------|-------|-------|
     *              feature map
     */
    // the proportion of feature map which needs to be loaded twice due to cut on m axis.
    // Up to now, only considered stride = 1 and each row overlaping only for ones, which may cause performace problem!
    float iOverlap = static_cast<float>((opInfo_.wShape5D.at(2) - 1) * (blockDim.mDim - 1)) / opInfo_.iShape5D.at(2);
    // feature map size per core
    dataPerCore.iSize = static_cast<float>(dataSize.iSize) * (1 + iOverlap) / blockDim.batchDim / blockDim.mDim;
    // weight size per core
    dataPerCore.wSize = static_cast<float>(dataSize.wSize) / blockDim.nDim;
    // out size per core
    dataPerCore.oSize = static_cast<float>(dataSize.oSize) / blockDim.batchDim / blockDim.nDim / blockDim.mDim;
}

/*
 * get block dim tiling at runtime
 * @param        tiling        tiling struct
 * @return       tiling        tiling struct with inferred block dim values
 */
bool FastTiling::GetBlockDimTiling(Tiling& tiling)
{
    BlockDimRange tilingRange;
    GetBlockDimRange(tilingRange);
    uint32_t iSize = byteForDtype_.at(opInfo_.aType) * static_cast<uint32_t>(GetEleNum(opInfo_.iShape5D));
    uint32_t wSize = byteForDtype_.at(opInfo_.bType) * static_cast<uint32_t>(GetEleNum(opInfo_.wShape5D));
    uint32_t oSize = byteForDtype_.at(opInfo_.cType) * static_cast<uint32_t>(GetEleNum(opInfo_.oShape5D));
    BlockDimSize blockDimSize = {iSize, wSize, oSize};
    uint32_t batchIndex = 0;
    uint32_t nIndex = 0;
    uint32_t mIndex = 0;
    uint32_t gIndex = 0;
    // total time cost per core = compute time per core + memory access time per core
    float estComputeTime = 0;
    float estMemoryAccTime = 0;
    float estTotalTime = FLT_MAX;

    // indicates which axis should be rewind after iteration
    uint32_t rewindIndicator = 0;
    BlockDimSize dataPerCore = {0, 0, 0};
    // iteration ended when block dim exceeds core number
    while (tilingRange.batchRange.at(batchIndex) * tilingRange.nRange.at(nIndex) *
           tilingRange.mRange.at(mIndex) * tilingRange.gRange.at(gIndex) <= hardware_.aicoreNum) {
        // estimate time cost per core under current block dim tiling strategy
        tiling.batchDim = tilingRange.batchRange.at(batchIndex);
        tiling.nDim = tilingRange.nRange.at(nIndex);
        tiling.mDim = tilingRange.mRange.at(mIndex);
        tiling.groupDim = tilingRange.gRange.at(gIndex);
        DataInOneCore(blockDimSize, tiling, dataPerCore);

        // compute amount(computing can parallel among cores) =
        // wSize(nk) * oSize(mn) /  wShape5D.at(0)(n) / blockDim = mnk / blockDim
        // compute time per core = datasize / cube_bandwidth + Featuremap prefusion cost + Featuremap postfusion cost
        auto cout = static_cast<uint32_t>(opInfo_.wShape5D.at(0));
        auto currentCoreNum = tiling.batchDim * tiling.nDim * tiling.mDim;
        estComputeTime = wSize * oSize / cout / currentCoreNum / hardware_.cubeBandwidth +
            dataPerCore.iSize / hardware_.vectorBandwidth * PREFUSION_TRANSPOSE_NUM +
            dataPerCore.wSize / hardware_.vectorBandwidth * POSTFUSION_TRANSPOSE_NUM;
        // meomry access time(cannot parallel as all memory access among cores share a common bus)
        // the summation of mte1, mte2, mte 3 time
        float mte2Inputs = dataPerCore.iSize * tiling.nDim / hardware_.l2Rate;
        float mte2Weights = dataPerCore.wSize * tiling.batchDim * tiling.mDim / hardware_.l2Rate;
        float mte2 = mte2Inputs + mte2Weights;
        float mte3 = dataPerCore.oSize / hardware_.ubToL2Rate;
        float mte1 = dataPerCore.iSize / hardware_.l1ToL0aRate + dataPerCore.wSize / hardware_.l1ToL0bRate +
                     dataPerCore.oSize * byteForDtype_.at(opInfo_.madType) / byteForDtype_.at(opInfo_.cType) /
                     hardware_.l0cToUbRate;
        estMemoryAccTime = mte1 + mte2 + mte3;
        float newEstTotalTime = estMemoryAccTime + estComputeTime;
        // iteration stop if cost function increase
        if (newEstTotalTime >= estTotalTime) {
            break;
        } else {
            estTotalTime = newEstTotalTime;
        }
        // tiling decisions
        if (gIndex < tilingRange.gRange.size() - 1) {
            // lift group dim to max as it will not affect total time cost
            gIndex++;
            rewindIndicator = (1 << FOUR);
        } else {
            if (dataPerCore.iSize > dataPerCore.wSize) {
                // cut the larger one between feature map and weight;
                // when cutting feature map, batch axis should be cut before m,
                // access as cutting m can bring redundant memory
                if (batchIndex < tilingRange.batchRange.size() - 1) {
                    // cut batch dim
                    batchIndex++;
                    rewindIndicator = (1 << 1);
                    continue;
                } else if (mIndex < tilingRange.mRange.size() - 1) {
                    mIndex++;
                    rewindIndicator = (1 << THREE);
                    continue;
                }
            }
            if (nIndex < tilingRange.nRange.size() - 1) {
                // cut n dim(cout dim)
                nIndex++;
                rewindIndicator = (1 << TWO);
            }
        }
    }
    // rewind tiling dims
    tiling.batchDim = tilingRange.batchRange.at(batchIndex - ((rewindIndicator >> 1) & 1));
    tiling.nDim = tilingRange.nRange.at(nIndex - ((rewindIndicator >> TWO) & 1));
    tiling.mDim = tilingRange.mRange.at(mIndex - ((rewindIndicator >> THREE) & 1));
    tiling.groupDim = tilingRange.gRange.at(gIndex - ((rewindIndicator >> FOUR) & 1));

    return true;
}

// get candidate values for every dims in block dim
void FastTiling::GetBlockDimRange(BlockDimRange& tilingRange)
{
    // obtain batch dim candidates
    CalcCommFactor(hardware_.aicoreNum, opInfo_.iShape5D.at(0), tilingRange.batchRange);
    // obtain n dim candidates
    CalcCommFactor(hardware_.aicoreNum, opInfo_.oShape5D.at(1), tilingRange.nRange);
    // obtain m dim candidates
    uint32_t alignedM = (opInfo_.oShape5D.at(2) * opInfo_.oShape5D.at(3) + getCi0() - 1) / getCi0();
    CalcCommFactor(hardware_.aicoreNum, std::ceil(alignedM), tilingRange.mRange);
    // obtain g dim candidates
    CalcCommFactor(hardware_.aicoreNum, opInfo_.groups, tilingRange.gRange);
}

/*
 * get the conv2d case status based on feature map/weight/result/mad_dtype parameters
 */
void FastTiling::GetConv2dCaseStatus()
{
    isQuantScene_ = (opInfo_.madType == ge::DataType::DT_INT32);
    // load2d case: fp16 case and kernel: 1x1, pad=0, stride=1
    isLoad2dFlag = ((opInfo_.aType == ge::DataType::DT_FLOAT16) &&
                    (opInfo_.bType == ge::DataType::DT_FLOAT16) &&
                    (opInfo_.kh == 1 && opInfo_.kw == 1) &&
                    (opInfo_.padl == 0 && opInfo_.padr == 0 && opInfo_.padu == 0 && opInfo_.padd == 0) &&
                    (opInfo_.stride_h == 1 && opInfo_.stride_w == 1) && opInfo_.hi != 1);

    isFmReadWithStrideFlag = (opInfo_.kh == 1) && (opInfo_.stride_h == 1) && (opInfo_.padl == 0)
                      && (opInfo_.padr == 0) && (opInfo_.padu == 0) && (opInfo_.padd == 0);
    isSplitWAxis_ = (!isLoad2dFlag) && (opInfo_.hi == 1)
                    && (opInfo_.kh == 1) && (opInfo_.padu == 0) && (opInfo_.padd == 0);
    khDilated_ = (opInfo_.kh - 1) * opInfo_.dilations_h + 1;
    kwDilated_ = (opInfo_.kw - 1) * opInfo_.dilations_w + 1;
    reduceKAxisAL1_KhDilKwDilCi0_ = getCi0() * khDilated_ * kwDilated_;
    reduceKAxisBL1_KhKwCi0_ = getCi0() * opInfo_.kh * opInfo_.kw;
}

// get L1 tiling every dim range
void FastTiling::GetL1TilingRange(Tiling &tiling)
{
    tilingRangeL1_.mAL1.clear();
    tilingRangeL1_.kAL1.clear();
    tilingRangeL1_.nBL1.clear();
    tilingRangeL1_.kBL1.clear();
    tilingRangeL1_.batch.clear();
    tilingRangeL1_.group.clear();

    // batch range
    vector<uint32_t> batchVector;
    uint32_t batchPerCore = (opInfo_.batch + tiling.batchDim -1) / tiling.batchDim;
    CalcCommFactor(batchPerCore, batchPerCore, batchVector);
    tilingRangeL1_.batch.assign(batchVector.begin(), batchVector.end());

    // nBL1 range
    vector<uint32_t> nBL1Vector;
    uint32_t nSizePerCore = (opInfo_.oShape5D[1] + tiling.nDim - 1) / tiling.nDim;
    CalcCommFactor(nSizePerCore, nSizePerCore, nBL1Vector);
    tilingRangeL1_.nBL1.assign(nBL1Vector.begin(), nBL1Vector.end());

    // mAL1 range
    vector<uint32_t> mAL1Vector;
    uint32_t mSize  =  (opInfo_.oShape5D[2] * opInfo_.oShape5D[3] + CUBE_UNIT_16 - 1) / CUBE_UNIT_16;
    uint32_t mSizePerCore = (mSize + tiling.mDim -1) / tiling.mDim;
    CalcCommFactor(MSIZE_PER_CORE, mSizePerCore, mAL1Vector);
    tilingRangeL1_.mAL1.assign(mAL1Vector.begin(), mAL1Vector.end());

    // kAL1 range
    vector<uint32_t> kAL1Vector;
    uint32_t kAL1Size = opInfo_.iShape5D.at(1); // 5hd[N,ci,Hi,Wi,c0]
    CalcCommFactor(kAL1Size, kAL1Size, kAL1Vector);
    // for reusing fmap, KAL1 * ReduceA > KBL1 * ReduceB
    vector<uint32_t>::iterator beginIter = kAL1Vector.end() - 1;
    uint32_t kAL1Min = kAL1Size * reduceKAxisBL1_KhKwCi0_ / reduceKAxisAL1_KhDilKwDilCi0_;
    for (auto iter = kAL1Vector.end() - 1; iter != kAL1Vector.begin(); iter--) {
        if (*iter >= kAL1Min) {
            beginIter = iter;
        }
    }
    // c1 > kAL1 > kAL1Min
    tilingRangeL1_.kAL1.assign(beginIter, kAL1Vector.end());

    // kBL1 range
    vector<uint32_t> kBL1Vector;
    uint32_t kBL1Size = opInfo_.iShape5D.at(1); // 5hd[N,ci,Hi,Wi,c0]
    CalcCommFactor(kBL1Size, kBL1Size, kBL1Vector);
    tilingRangeL1_.kBL1.assign(kBL1Vector.end() - 1, kBL1Vector.end()); // fix kBL1 as Ci

    // group range
    vector<uint32_t> groupVector;
    uint32_t groupPerCore = (opInfo_.groups + tiling.groupDim -1) / tiling.groupDim;
    CalcCommFactor(groupPerCore, groupPerCore, groupVector);
    tilingRangeL1_.group.assign(groupVector.begin(), groupVector.end());
}

void FastTiling::UpdateL1Data()
{
    uint32_t al1HixWi;
    uint32_t kAL1 = tilingRangeL1_.kAL1.at(l1Data_.kAL1_index);
    uint32_t kBL1 = tilingRangeL1_.kBL1.at(l1Data_.kBL1_index);
    uint32_t mAL1 = tilingRangeL1_.mAL1.at(l1Data_.mAL1_index);
    uint32_t nBL1 = tilingRangeL1_.nBL1.at(l1Data_.nBL1_index);
    if (isSplitWAxis_) {
        uint32_t aL1Wo = (mAL1 * CUBE_UNIT_16 + opInfo_.oShape5D.at(3) -1) / opInfo_.oShape5D.at(3);
        uint32_t aL1Wi = aL1Wo * opInfo_.stride_w + kwDilated_;
        al1HixWi =  opInfo_.hi * aL1Wi;
    } else {
        uint32_t aL1Ho = (mAL1 * CUBE_UNIT_16 + opInfo_.oShape5D.at(3) -1) / opInfo_.oShape5D.at(3);
        uint32_t aL1Hi = aL1Ho * opInfo_.stride_h +  khDilated_;
        if (aL1Hi > opInfo_.hi) {
            aL1Hi = opInfo_.hi;
        }
        al1HixWi = (isFmReadWithStrideFlag ?
            ((aL1Hi + opInfo_.stride_h -1) / opInfo_.stride_h) : aL1Hi) * opInfo_.wi;
    }
    l1Data_.fmCurrent = kAL1 * CUBE_UNIT_16 * al1HixWi * byteForDtype_.at(opInfo_.aType);
    l1Data_.filter_current = kBL1 * getCi0() * opInfo_.kh * opInfo_.kw * nBL1 * CUBE_UNIT_16 * byteForDtype_.at(opInfo_.bType);
    l1Data_.l1_current = l1Data_.fmCurrent + l1Data_.result_current;
}

uint32_t FastTiling::getCi0()
{
    if (isQuantScene_) {
        return CUBE_UNIT_32;
    } else {
        return CUBE_UNIT_16;
    }
}

int64_t FastTiling::GetEleNum(const vector<int64_t>& shape)
{
    int64_t eleNum = 1;
    for (const auto &dim : shape) {
        eleNum *= dim;
    }
    return eleNum;
}

void FastTiling::AddKbl1()
{
    if (l1Data_.l1_current < hardware_.l1Size / TWO &&
        l1Data_.kBL1_index < (tilingRangeL1_.kBL1.size() -1) &&
        (l1Data_.fmCurrent + l1Data_.filter_current / tilingRangeL1_.kBL1.at(l1Data_.kBL1_index) *
         tilingRangeL1_.kBL1.at(l1Data_.kBL1_index + 1)) < hardware_.l1Size / TWO) {
        l1Data_.kBL1_index++;
    }
}

void FastTiling::AddKal1()
{
    if (l1Data_.l1_current < hardware_.l1Size / TWO &&
        l1Data_.kAL1_index < (tilingRangeL1_.kAL1.size() -1) &&
        (l1Data_.filter_current + l1Data_.fmCurrent / tilingRangeL1_.kBL1.at(l1Data_.kBL1_index) *
         tilingRangeL1_.kBL1.at(l1Data_.kBL1_index + 1)) < hardware_.l1Size / TWO) {
        l1Data_.kAL1_index++;
    }
}

void FastTiling::AddMal1()
{
    if (l1Data_.l1_current < hardware_.l1Size / TWO &&
        l1Data_.mAL1_index < (tilingRangeL1_.mAL1.size() -1) &&
        (l1Data_.filter_current + l1Data_.fmCurrent / tilingRangeL1_.mAL1.at(l1Data_.mAL1_index) *
         tilingRangeL1_.mAL1.at(l1Data_.mAL1_index + 1)) < hardware_.l1Size / TWO) {
        l1Data_.mAL1_index++;
    }
}

void FastTiling::AddNbl1()
{
    if (l1Data_.l1_current < hardware_.l1Size / TWO &&
        l1Data_.nBL1_index < (tilingRangeL1_.nBL1.size() -1) &&
        (l1Data_.fmCurrent + l1Data_.filter_current / tilingRangeL1_.nBL1.at(l1Data_.nBL1_index) *
         tilingRangeL1_.nBL1.at(l1Data_.nBL1_index + 1)) < hardware_.l1Size / TWO) {
        l1Data_.nBL1_index++;
    }
}

bool JudgeL1(const pair<string, float> numberA, const pair<string, float> numberB) {
    return numberA.second < numberB.second;
}

void FastTiling::AddL1Data(const pair<string, float> data) {
    if (data.first == "nBL1") {
        AddNbl1();
    } else if (data.first == "mAL1") {
        AddMal1();
    } else if (data.first == "kAL1") {
        AddKal1();
    } else { // "kBL1'"
        AddKbl1();
    }
}

bool FastTiling::GetL1Tiling(Tiling &tiling)
{
    // initialize
    GetL1TilingRange(tiling);
    float nBL1scale = hardware_.l0cToUbRate / hardware_.l2Rate;
    float mAL1scale = hardware_.l0cToUbRate / hardware_.l2Rate;
    float kAL1scale = hardware_.l2Rate / hardware_.l0cToUbRate;
    float kBL1scale = hardware_.l2Rate / hardware_.l0cToUbRate;
    std::vector<pair<string, float>> l1TempVector;
    uint32_t pre_mAL1_index = -1;
    uint32_t pre_nBL1_index = -1;
    uint32_t pre_kAL1_index = -1;
    uint32_t pre_kBL1_index = -1;

    UpdateL1Data();
    // get l1 tiling
    // target function min(1/l2Rate*nBL1  + 1/l2Rate*mAL1 + 1/l0cToUbRate*kal1 + 1/l0cToUbRate*kbl1)
    while (l1Data_.l1_current <=  hardware_.l1Size / TWO) {
        // ping pong is on by default
        bool breakFlag = (l1Data_.nBL1_index == pre_nBL1_index) && (l1Data_.mAL1_index == pre_mAL1_index)
                          && (l1Data_.kAL1_index == pre_kAL1_index) && (l1Data_.kBL1_index == pre_kBL1_index);
        if (breakFlag) {
            break;
        } else {
            pre_nBL1_index = l1Data_.nBL1_index;
            pre_mAL1_index = l1Data_.mAL1_index;
            pre_kAL1_index = l1Data_.kAL1_index;
            pre_kBL1_index = l1Data_.kBL1_index;
        }

        float nBL1number = nBL1scale * tilingRangeL1_.nBL1.at(l1Data_.nBL1_index);
        float mAL1number = mAL1scale * tilingRangeL1_.mAL1.at(l1Data_.mAL1_index);
        float kAL1number = kAL1scale * tilingRangeL1_.kAL1.at(l1Data_.kAL1_index);
        float kBL1number = kBL1scale * tilingRangeL1_.kBL1.at(l1Data_.kBL1_index);
        l1TempVector.push_back(make_pair("nBL1", nBL1number));
        l1TempVector.push_back(make_pair("mAL1", mAL1number));
        l1TempVector.push_back(make_pair("kAL1", kAL1number));
        l1TempVector.push_back(make_pair("kBL1", kBL1number));

        sort(l1TempVector.begin(), l1TempVector.end(), JudgeL1);
        for (auto data : l1TempVector) {
            AddL1Data(data);
        }
        UpdateL1Data();
        l1TempVector.clear();
    }
    uint32_t kBL1 = tilingRangeL1_.kBL1.at(l1Data_.kBL1_index);
    uint32_t nBL1 = tilingRangeL1_.nBL1.at(l1Data_.nBL1_index);
    l1Data_.filter_current = kBL1 * getCi0() * opInfo_.kh * opInfo_.kw * nBL1 * CUBE_UNIT_16 * byteForDtype_.at(opInfo_.bType);
    // assignment
    AssignmentL1(tiling);
    return true;
}

void FastTiling::AssignmentL1(Tiling& tiling) {
    if (tilingRangeL1_.kBL1.at(l1Data_.kBL1_index) == opInfo_.iShape5D.at(1) &&
        tilingRangeL1_.nBL1.at(l1Data_.nBL1_index) == opInfo_.oShape5D.at(1)) {
        if (l1Data_.filter_current <= hardware_.l0bSize / TWO && tiling.kAL1 == opInfo_.iShape5D.at(1)) {
        // filter full load to l0B
            tiling.nBL1 = 0;
            tiling.kBL1 = 0;
        } else {
            tiling.nBL1 = FULL_LOAD;
            tiling.kBL1 = FULL_LOAD;
        }
    } else {
        tiling.nBL1Value = tilingRangeL1_.nBL1.at(l1Data_.nBL1_index);
        tiling.kBL1ci = tilingRangeL1_.kBL1.at(l1Data_.kBL1_index);
        tiling.kBL1 = tiling.kBL1ci * reduceKAxisBL1_KhKwCi0_;
    }

    tiling.mAL1Value = tilingRangeL1_.mAL1.at(l1Data_.mAL1_index);
    tiling.kAL1ci = tilingRangeL1_.kAL1.at(l1Data_.kAL1_index);
    tiling.kAL1 = tiling.kAL1ci * reduceKAxisAL1_KhDilKwDilCi0_;
}

/**
 * calculate the common factor of an unsigned int value
 * @param [in] num                   input unsigned int value
 * @param [in] numMax                input max of unsigned int value
 * @return     rlist                 vector, contain the common factors
 */
void FastTiling::CalcCommFactor(const uint32_t num, const uint32_t numMax, std::vector<uint32_t>& rlist)
{
    if (num == 0) {
        rlist.push_back(0);
    } else if (num == 1) {
        rlist.push_back(1);
    } else {
        uint32_t i = 1; // calculate the common factor, exclude 0;
        while (i != 0 && i <= numMax) {
            if (num % i == 0) {
                rlist.push_back(i);
            }
            i++;
        }
    }
}

void FastTiling::UpdateL0Data()
{
    uint32_t kL0 = tilingRangeL0_.kL0.at(l0Data_.kL0Index);
    uint32_t mL0 = tilingRangeL0_.mL0.at(l0Data_.mL0Index);
    uint32_t nL0 = tilingRangeL0_.nL0.at(l0Data_.nL0Index);

    // batch and group need to consider in future.
    l0Data_.l0ACurrent = kL0 * getCi0() * opInfo_.kh * opInfo_.kw * mL0 * CUBE_UNIT_16 * byteForDtype_.at(opInfo_.aType);
    l0Data_.l0BCurrent = kL0 * getCi0() * opInfo_.kh * opInfo_.kw * nL0 * CUBE_UNIT_16 * byteForDtype_.at(opInfo_.bType);
    l0Data_.l0CCurrent = mL0 * nL0 * CUBE_UNIT_16 * getCi0() *  byteForDtype_.at(opInfo_.cType);
}

// get every dim range in L0 tiling
void FastTiling::GetL0TilingRange(const Tiling& tiling)
{
    tilingRangeL0_.mL0.clear();
    tilingRangeL0_.kL0.clear();
    tilingRangeL0_.kAL0.clear();
    tilingRangeL0_.kBL0.clear();
    tilingRangeL0_.nL0.clear();
    tilingRangeL0_.groupL0.clear();
    tilingRangeL0_.batchL0.clear();

    // get l0 batch range
    vector<uint32_t> batchVectorL0;
    CalcCommFactor(tiling.batchAL1, tiling.batchAL1, batchVectorL0);
    tilingRangeL0_.batchL0.assign(batchVectorL0.begin(), batchVectorL0.end());

    // get l0 group range
    vector<uint32_t> groupVectorL0;
    CalcCommFactor(tiling.groupAL1, tiling.groupAL1, groupVectorL0);
    tilingRangeL0_.groupL0.assign(groupVectorL0.begin(), groupVectorL0.end());

    // get l0 m range
    vector<uint32_t> mVectorL0;
    CalcCommFactor(tiling.mAL1Value, tiling.mAL1Value, mVectorL0);
    // no Load2dFlag case, mL0 can noly be even or 1
    // to do: only support range Power of 2 to reduce the range space
    if (!isLoad2dFlag) {
        for (auto it : mVectorL0) {
            if (it % TWO == 0) {
                tilingRangeL0_.mL0.push_back(it);
            }
        }
        if (tilingRangeL0_.mL0.empty()) {
            tilingRangeL0_.mL0.push_back(1);
        }
    } else {
        tilingRangeL0_.mL0.assign(mVectorL0.begin(), mVectorL0.end());
    }

    vector<uint32_t> nVectorL0;
    vector<uint32_t> kAVectorL0;
    vector<uint32_t> kBVectorL0;
    uint32_t nMaxAvail = (opInfo_.oShape5D[1] + tiling.nDim -1) / tiling.nDim;
    uint32_t kBMaxAvail = opInfo_.iShape5D[1];
    if (tiling.nBL1 == FULL_LOAD) {
        // weight can load from ddr to l1 once, but can not load from l1 to l0.
        // when weight is FULL_LOAD L1, get n,k range from ori shape per core
        CalcCommFactor(nMaxAvail, nMaxAvail, nVectorL0);
        tilingRangeL0_.nL0.assign(nVectorL0.begin(), nVectorL0.end());

        CalcCommFactor(tiling.kAL1ci, tiling.kAL1ci, kAVectorL0);
        CalcCommFactor(kBMaxAvail, kBMaxAvail, kBVectorL0);
        tilingRangeL0_.kAL0.assign(kAVectorL0.begin(), kAVectorL0.end());
        tilingRangeL0_.kBL0.assign(kBVectorL0.begin(), kBVectorL0.end());
    } else if (tiling.nBL1 == 0) {
        // weight can load from ddr to l1 once, and can load from l1 to l0 once -->
        // weight load from ddr to l0, directly.
        // when weight is FULL_LOAD DDR -> L0B, n,k range equal to maxAvail
        tilingRangeL0_.nL0.push_back(nMaxAvail);

        CalcCommFactor(tiling.kAL1ci, tiling.kAL1ci, kAVectorL0);
        tilingRangeL0_.kAL0.assign(kAVectorL0.begin(), kAVectorL0.end());
        tilingRangeL0_.kBL0.push_back(kBMaxAvail);
    } else {
        CalcCommFactor(tiling.nBL1Value, tiling.nBL1Value, nVectorL0);
        tilingRangeL0_.nL0.assign(nVectorL0.begin(), nVectorL0.end());

        CalcCommFactor(tiling.kAL1ci, tiling.kAL1ci, kAVectorL0);
        CalcCommFactor(tiling.kBL1ci, tiling.kBL1ci, kBVectorL0);
        tilingRangeL0_.kAL0.assign(kAVectorL0.begin(), kAVectorL0.end());
        tilingRangeL0_.kBL0.assign(kBVectorL0.begin(), kBVectorL0.end());
    }

    if (tiling.nBL1 == 0) {
        tilingRangeL0_.kL0.assign(tilingRangeL0_.kAL0.begin(), tilingRangeL0_.kAL0.end());
    } else {
        for (vector<uint32_t>::const_iterator kIterator = tilingRangeL0_.kAL0.begin();
             kIterator != tilingRangeL0_.kAL0.end(); kIterator++) {
            if (find(tilingRangeL0_.kBL0.begin(), tilingRangeL0_.kBL0.end(), *kIterator) != tilingRangeL0_.kBL0.end()) {
                tilingRangeL0_.kL0.push_back(*kIterator);
            }
        }
    }
}

void FastTiling::WeightFullLoad()
{
    // first add mAL0, secondly add kAL0, cause mAL0 related to task reused.
    // when weight full_load from ddr -> L0B, only decision input feature map mAL0 and kAL0
    while (l0Data_.l0ACurrent <= hardware_.l0aSize / tilingRangeL0_.pbAL0.at(0) &&
        l0Data_.l0CCurrent <= hardware_.l0cSize / tilingRangeL0_.pbCL0.at(0)) {
        if (!l0Data_.updateFlag) {
            break;
        }
        l0Data_.updateFlag = false;
        AddmL0();
    }
    if (l0Data_.mL0Index == tilingRangeL0_.mL0.size() - 1) {
        // tmpKAIndex 0, calculate before.
        for (uint32_t tmpKAIndex = 1; tmpKAIndex < tilingRangeL0_.kL0.size(); tmpKAIndex++) {
            uint32_t tmpL0ACurrent = tilingRangeL0_.kL0.at(tmpKAIndex) * getCi0() * opInfo_.kh *
                                    opInfo_.kw * tilingRangeL0_.mL0.at(l0Data_.mL0Index) *
                                    CUBE_UNIT_16 * byteForDtype_.at(opInfo_.aType);
            if (tmpL0ACurrent > hardware_.l0aSize / tilingRangeL0_.pbAL0.at(0)) {
                break;
            }
            l0Data_.kL0Index = tmpKAIndex;
        }
    }
}

// calcualte L0 Tiling
bool FastTiling::GetL0Tiling(Tiling& tiling)
{
    // initialize
    GetL0TilingRange(tiling);
    UpdateL0Data();

    if (tiling.nBL1 == 0) {
        WeightFullLoad();
    } else {
        WeightL1ToL0();
        WeightL1ToL0Reset(tiling);
    }
    AssignmentL0(tiling);
    return true;
}

bool FastTiling::L1ToL0SizeSatify()
{
    if (l0Data_.l0ACurrent <= hardware_.l0aSize / tilingRangeL0_.pbAL0.at(0) &&
        l0Data_.l0BCurrent <= hardware_.l0bSize / tilingRangeL0_.pbBL0.at(0) &&
        l0Data_.l0CCurrent <= hardware_.l0cSize / tilingRangeL0_.pbCL0.at(0)) {
        return true;
    }
    return false;
}

void FastTiling::WeightL1ToL0()
{
    uint32_t L0Bscale = hardware_.l1ToL0bRate;
    uint32_t L0Ascale = hardware_.l1ToL0aRate;
    uint32_t L0Cscale = hardware_.l0cToUbRate;
    uint32_t mTmp = tilingRangeL0_.mL0.at(l0Data_.mL0Index);
    uint32_t nTmp = tilingRangeL0_.nL0.at(l0Data_.nL0Index);
    uint32_t kTmp = tilingRangeL0_.kL0.at(l0Data_.kL0Index);
    // target function = 1/(l1ToL0aRate*nL0) + 1/(l1ToL0bRate*mL0) + 1/(l0cToUbRate*kL0)
    while (L1ToL0SizeSatify()) {
        if (!l0Data_.updateFlag) {
            break;
        }
        l0Data_.updateFlag = false;
        if (nTmp * L0Ascale <= mTmp * L0Bscale && kTmp * L0Cscale <= mTmp * L0Bscale) {
            if (kTmp * L0Cscale < nTmp * L0Ascale) {
                // m > n > k in ratio
                AddkL0();
                AddnL0();
                AddmL0();
            } else {
                // m > k > n in ratio
                AddnL0();
                AddkL0();
                AddmL0();
            }
        } else if (nTmp * L0Ascale >= mTmp * L0Bscale && kTmp * L0Cscale <= nTmp * L0Ascale) {
            if (mTmp * L0Bscale < kTmp * L0Cscale) {
                // n > k > m in ratio
                AddmL0();
                AddkL0();
                AddnL0();
            } else {
                AddkL0();
                AddmL0();
                AddnL0();
            }
        } else {
            if (mTmp * L0Bscale < nTmp * L0Ascale) {
                // k > n > m in ratio
                AddmL0();
                AddnL0();
                AddkL0();
            } else {
                // k > m > n in ratio
                AddnL0();
                AddmL0();
                AddkL0();
            }
        }
    }
}

void FastTiling::WeightL1ToL0Reset(Tiling &tiling)
{
    // when n/kL0 == n/kL1, load weight from ddr to l0B, cut up.
    if (tilingRangeL0_.nL0.at(l0Data_.nL0Index) == tiling.nBL1Value &&
        tilingRangeL0_.kL0.at(l0Data_.kL0Index) == tiling.kBL1ci) {
        // also means nBl1 zero.
        tiling.nBL1Value = 0;
        tiling.kBL1ci = 0;
        // tiling.kBL1 is tiling.kBL1ci * reduceKAxisAL1_KhDilKwDilCi0_
        tiling.kBL1 = 0;
        // weight extra ->(/k) m ->(/)
        uint32_t mAExtraSpace = l0Data_.l0BCurrent / reduceKAxisAL1_KhDilKwDilCi0_ /
                                tiling.kAL1ci / tilingRangeL0_.mL0.at(l0Data_.mL0Index);
        tiling.mAL1Value += mAExtraSpace * tilingRangeL0_.mL0.at(l0Data_.mL0Index);
    }
}

void FastTiling::AssignmentL0(Tiling &tiling)
{
    // set tiling L0
    uint32_t reduceKAxisL0_KhKwCi0_ = getCi0() * opInfo_.kh * opInfo_.kw;
    tiling.kA = tilingRangeL0_.kL0.at(l0Data_.kL0Index) * reduceKAxisL0_KhKwCi0_ / getCi0();
    tiling.mA = tilingRangeL0_.mL0.at(l0Data_.mL0Index);
    tiling.kB = tiling.nBL1 == 0 ? FULL_LOAD : tiling.kA;
    tiling.nB = tiling.nBL1 == 0 ? FULL_LOAD : tilingRangeL0_.nL0.at(l0Data_.nL0Index);
    tiling.mC = tiling.mA;
    if (tiling.nB != FULL_LOAD) {
        tiling.nC = tiling.nB;
    } else {
        // when nB is FULL_LOAD, calculate nc
        tiling.nC = tilingRangeL0_.nL0.at(l0Data_.nL0Index);
    }
    // special case, set L1.
    tiling.mAL1 = tiling.mAL1Value / tiling.mA;
    // nBL1 is FULL_LOAD, pass it to schedule.
    if (tiling.nBL1 != FULL_LOAD) {
        tiling.nBL1 = tiling.nBL1Value == 0 ? 0 : tiling.nBL1Value / tiling.nB;
    }
}

void FastTiling::AddkL0()
{
    if (l0Data_.kL0Index < (tilingRangeL0_.kL0.size() - 1)) {
        if (l0Data_.l0ACurrent / tilingRangeL0_.kL0.at(l0Data_.kL0Index) *
            tilingRangeL0_.kL0.at(l0Data_.kL0Index + 1) <= hardware_.l0aSize / tilingRangeL0_.pbAL0.at(0) &&
            l0Data_.l0BCurrent / tilingRangeL0_.kL0.at(l0Data_.kL0Index) *
            tilingRangeL0_.kL0.at(l0Data_.kL0Index + 1) <= hardware_.l0bSize / tilingRangeL0_.pbBL0.at(0)) {
            // update kL0
            l0Data_.kL0Index++;
            l0Data_.updateFlag = true;
            UpdateL0Data();
        }
    }
}

void FastTiling::AddnL0()
{
    if (l0Data_.nL0Index < (tilingRangeL0_.nL0.size() - 1)) {
        if (l0Data_.l0BCurrent / tilingRangeL0_.nL0.at(l0Data_.nL0Index) *
            tilingRangeL0_.nL0.at(l0Data_.nL0Index + 1) <= hardware_.l0bSize / tilingRangeL0_.pbBL0.at(0) &&
            l0Data_.l0CCurrent / tilingRangeL0_.nL0.at(l0Data_.nL0Index) *
            tilingRangeL0_.nL0.at(l0Data_.nL0Index + 1) <= hardware_.l0cSize / tilingRangeL0_.pbCL0.at(0)) {
            // update nL0
            l0Data_.nL0Index++;
            l0Data_.updateFlag = true;
            UpdateL0Data();
        }
    }
}

void FastTiling::AddmL0()
{
    if (l0Data_.mL0Index < (tilingRangeL0_.mL0.size() - 1)) {
        if (l0Data_.l0ACurrent / tilingRangeL0_.mL0.at(l0Data_.mL0Index) *
            tilingRangeL0_.mL0.at(l0Data_.mL0Index + 1) <= hardware_.l0aSize / tilingRangeL0_.pbAL0.at(0) &&
            l0Data_.l0CCurrent / tilingRangeL0_.mL0.at(l0Data_.mL0Index) *
            tilingRangeL0_.mL0.at(l0Data_.mL0Index + 1) <= hardware_.l0cSize / tilingRangeL0_.pbAL0.at(0)) {
            // update mL0
            l0Data_.mL0Index++;
            l0Data_.updateFlag = true;
            UpdateL0Data();
        }
    }
}

void FastTiling::UpdateUBData()
{
    uint32_t kAub = tilingRangeUB_.kAub.at(ubData_.kAubIndex);
    uint32_t mAub = tilingRangeUB_.mAub.at(ubData_.mAubIndex);
    uint32_t ncFactor = tilingRangeUB_.ncFactor.at(ubData_.ncFactorIndex);
    uint32_t mcFactor = tilingRangeUB_.mcFactor.at(ubData_.mcFactorIndex);

    uint32_t tmpPreUbCurrent = kAub * reduceKAxisAL1_KhDilKwDilCi0_ * mAub
                               * CUBE_UNIT_16 * byteForDtype_.at(opInfo_.aType);
    uint32_t tmpPostUbCurrent = ncFactor * mcFactor * 256;

    ubData_.preUbCurrent = opInfo_.preFusionUbUtilize * tmpPreUbCurrent;
    ubData_.postUbCurrent = opInfo_.postFusionUbUtilize * tmpPostUbCurrent;
}

// get every dim range in UB tiling
void FastTiling::GetUBTilingRange(const Tiling& tiling)
{
    tilingRangeUB_.kAub.clear();
    tilingRangeUB_.mAub.clear();
    tilingRangeUB_.ncFactor.clear();
    tilingRangeUB_.mcFactor.clear();

    // get UB kAub range
    vector<uint32_t> kAubVector;
    CalcCommFactor(tiling.kAL1ci, tiling.kAL1ci, kAubVector); // 待schedule确认新含义
    tilingRangeUB_.kAub.assign(kAubVector.begin(), kAubVector.end());

    // get UB mAub range
    vector<uint32_t> mAubVector;
    CalcCommFactor(tiling.mAL1Value, tiling.mAL1Value, mAubVector);
    tilingRangeUB_.mAub.assign(mAubVector.begin(), mAubVector.end());

    // get UB ncFactor range
    vector<uint32_t> ncFactorVector;
    CalcCommFactor(tiling.nC, tiling.nC, ncFactorVector);
    tilingRangeUB_.ncFactor.assign(ncFactorVector.begin(), ncFactorVector.end());

    // get UB mcFactor range
    tilingRangeUB_.mcFactor.push_back(tiling.mC);
}

// calcualte L0 Tiling
bool FastTiling::GetUBTiling(Tiling& tiling)
{
    // initialize
    GetUBTilingRange(tiling);
    UpdateUBData();
    // get tiling front fusion
    while (ubData_.preUbCurrent <= hardware_.ubSize) {
        if (!ubData_.updateFlag) {
            break;
        }
        ubData_.updateFlag = false;
        AddmAub();
        AddkAub();
    }

    // get tiling post fusion
    for (uint32_t ncFactorIndex = 0; ncFactorIndex < tilingRangeUB_.ncFactor.size() - 1; ncFactorIndex++) {
        uint32_t postUBCurrent = tilingRangeUB_.ncFactor.at(ncFactorIndex + 1) *
                                 tilingRangeUB_.mcFactor.at(ubData_.mcFactorIndex) * 256 *
                                 opInfo_.postFusionUbUtilize;
        if (postUBCurrent <= hardware_.ubSize) {
            ubData_.ncFactorIndex = ncFactorIndex + 1;
        } else {
            break;
        }
    }
    // set Ub tiling decision
    tiling.kAub = tilingRangeUB_.kAub.at(ubData_.kAubIndex);
    tiling.mAub = tilingRangeUB_.mAub.at(ubData_.mAubIndex);
    tiling.nCFactor = tilingRangeUB_.ncFactor.at(ubData_.ncFactorIndex);
    tiling.mCFactor = tiling.mC;
    return true;
}

void FastTiling::AddmAub()
{
    if (ubData_.mAubIndex < (tilingRangeUB_.mAub.size() - 1)) {
        if (ubData_.preUbCurrent / tilingRangeUB_.mAub.at(ubData_.mAubIndex) *
            tilingRangeUB_.mAub.at(ubData_.mAubIndex + 1) <= hardware_.ubSize) {
            // update mAub
            ubData_.mAubIndex++;
            ubData_.updateFlag = true;
            UpdateUBData();
        }
    }
}

void FastTiling::AddkAub()
{
    if (ubData_.kAubIndex < (tilingRangeUB_.kAub.size() - 1)) {
        if (ubData_.preUbCurrent / tilingRangeUB_.kAub.at(ubData_.kAubIndex) *
            tilingRangeUB_.kAub.at(ubData_.kAubIndex + 1) <= hardware_.ubSize) {
            // update kAub
            ubData_.kAubIndex++;
            ubData_.updateFlag = true;
            UpdateUBData();
        }
    }
}
}
