#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "array_ops.h"
#define private public
#include "op_tiling/conv2d_fast_tiling.h"

using namespace optiling;

class Conv2DFastTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Conv2DFastTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Conv2DFastTilingTest TearDown" << std::endl;
    }

    virtual void SetUp() {
        inputParams.batch = 1;
        inputParams.fmci = 32;
        inputParams.hi = 32;
        inputParams.wi = 32;
        inputParams.n = 1;
        inputParams.wci = 32;
        inputParams.kh = 3;
        inputParams.kw = 3;
        inputParams.ho = 30;
        inputParams.wo = 30;

        hardwareInfo.aicoreNum = 48;
        hardwareInfo.l2Size = 33554432;
        hardwareInfo.l1Size = 524288;
        hardwareInfo.l0aSize = 65536;
        hardwareInfo.l0bSize = 65536;
        hardwareInfo.l0cSize = 131072;
        hardwareInfo.ubSize = 196608;
        hardwareInfo.btSize = 1024;
        hardwareInfo.ddrReadRate = 32;
        hardwareInfo.ddrWriteRate = 32;
        hardwareInfo.l2Rate = 110;
        hardwareInfo.l2ReadRate = 110;
        hardwareInfo.l2WriteRate = 86;
        hardwareInfo.l1ToL0aRate = 512;
        hardwareInfo.l1ToL0bRate = 256;
        hardwareInfo.l1ToUbRate = 128;
        hardwareInfo.l0cToUbRate = 256;
        hardwareInfo.ubToL2Rate = 64;
        hardwareInfo.ubToDdrRate = 64;
        hardwareInfo.ubToL1Rate = 128;
        std::cout << "SetUp" << std::endl;
    }

    virtual void TearDown() {
         std::cout << "TearDown" << std::endl;

    }
    Conv2dParams inputParams;
    HardwareInfo hardwareInfo;
    Conv2dTiling tiling; 
};

TEST_F(Conv2DFastTilingTest, test_get_l1_tiling_default) {
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    Tiling generalTiling;
    bool ret = fastTilingPtr->GetL1Tiling(generalTiling);
    ASSERT_TRUE(ret);
}

TEST_F(Conv2DFastTilingTest, test_get_l1_tiling_split_axis) {
    inputParams.hi = 1;
    inputParams.kh = 1;
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    Tiling generalTiling;
    bool ret = fastTilingPtr->GetL1Tiling(generalTiling);
    ASSERT_TRUE(ret);
}

TEST_F(Conv2DFastTilingTest, test_get_l1_full_load_l1) {
    // small weight full load to L1
    hardwareInfo.l0bSize = 10000;
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    Tiling generalTiling;
    bool ret = fastTilingPtr->GetL1Tiling(generalTiling);
    ASSERT_TRUE(ret);
}

TEST_F(Conv2DFastTilingTest, test_get_l1_large_weight) {
    // large weight to tile
    inputParams.hi = 128;
    inputParams.wi = 128;
    inputParams.kh = 16;
    inputParams.kw = 16;
    inputParams.ho = 112;
    inputParams.wo = 112;
    inputParams.n = 128; //cout
    hardwareInfo.l1Size = 524288 * 2;
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    Tiling generalTiling;
    bool ret = fastTilingPtr->GetL1Tiling(generalTiling);
    ASSERT_TRUE(ret);
}

TEST_F(Conv2DFastTilingTest, test_get_blockDim_tiling) {
    unique_ptr<FastTiling> blockTilingPtr(new FastTiling());
    inputParams.batch = 32;
    blockTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling generalTiling;
    bool ret = blockTilingPtr->GetBlockDimTiling(generalTiling);
    ASSERT_EQ(true, ret);
}


TEST_F(Conv2DFastTilingTest, test_get_blockDim_tiling_group) {
    unique_ptr<FastTiling> blockTilingPtr(new FastTiling());
    inputParams.batch = 32;
    inputParams.groups = 2;
    blockTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling generalTiling;
    bool ret = blockTilingPtr->GetBlockDimTiling(generalTiling);
    ASSERT_EQ(true, ret);
}

TEST_F(Conv2DFastTilingTest, test_get_blockDim_tiling_Ashape_larger_Bshape) {
    unique_ptr<FastTiling> blockTilingPtr(new FastTiling());
    inputParams.batch = 32;
    inputParams.hi = 128;
    inputParams.wi = 128;
    inputParams.fmci = 2048;
    inputParams.kh = 16;
    inputParams.kw = 16;
    inputParams.wci = 3;
    inputParams.ho = 112;
    inputParams.wo = 112;
    blockTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling generalTiling;
    bool ret = blockTilingPtr->GetBlockDimTiling(generalTiling);
    ASSERT_EQ(true, ret);
}

TEST_F(Conv2DFastTilingTest, test_calc_common_factor_special_case)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    const uint32_t num_zero = 0;
    const uint32_t numMax = 0;
    std::vector<uint32_t> rlist;
    fastTilingPtr->CalcCommFactor(num_zero, numMax, rlist);
    ASSERT_EQ(rlist.size(), 1);
    ASSERT_EQ(rlist[0], 0);

    rlist.clear();
    const uint32_t num_one = 1;
    fastTilingPtr->CalcCommFactor(num_one, numMax, rlist);
    ASSERT_EQ(rlist.size(), 1);
    ASSERT_EQ(rlist[0], 1);
}

TEST_F(Conv2DFastTilingTest, test_calc_common_factor)
{
    unique_ptr<FastTiling> fastTlingPtr(new FastTiling());
    const uint32_t num = 8;
    const uint32_t numMax = 5;
    std::vector<uint32_t> rlist;
    std::vector<uint32_t> res = {1, 2, 4};
    fastTlingPtr->CalcCommFactor(num, numMax, rlist);
    ASSERT_EQ(rlist.size(), res.size());
    for (int idx = 0; idx < rlist.size(); idx++) {
        ASSERT_EQ(rlist[idx], res[idx]);
    }
}

TEST_F(Conv2DFastTilingTest, test_update_l0_data)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    // You may need to check the index in source file.
    fastTilingPtr->tilingRangeL0_.kL0 = {1};
    fastTilingPtr->tilingRangeL0_.mL0 = {2};
    fastTilingPtr->tilingRangeL0_.nL0 = {3};
    // index default is 0 and get Ci0 is 16.
    uint32_t l0ACurrent = 1 * 16 * inputParams.kh * inputParams.kw * 2 * 16 * 2;
    uint32_t l0BCurrent = 1 * 16 * inputParams.kh * inputParams.kw * 3 * 16 * 2;
    uint32_t l0CCurrent = 2 * 3 * 16 * 16 * 2;
    fastTilingPtr->UpdateL0Data();
    ASSERT_EQ(fastTilingPtr->l0Data_.l0ACurrent, l0ACurrent);
    ASSERT_EQ(fastTilingPtr->l0Data_.l0BCurrent, l0BCurrent);
    ASSERT_EQ(fastTilingPtr->l0Data_.l0CCurrent, l0CCurrent);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_range)
{
    unique_ptr<FastTiling> fastTillingPtr(new FastTiling());
    fastTillingPtr->SetInputParams(inputParams, hardwareInfo);
    Tiling fake_tiling;
    fake_tiling.batchAL1 = 1;
    fake_tiling.groupAL1 = 1;
    fake_tiling.mAL1Value = 2;
    fake_tiling.nDim = 1;

    fake_tiling.nBL1 = FULL_LOAD;
    fastTillingPtr->GetL0TilingRange(fake_tiling);

    fake_tiling.nBL1 = 0;
    fastTillingPtr->GetL0TilingRange(fake_tiling);

    fake_tiling.mAL1Value = 3;
    fastTillingPtr->GetL0TilingRange(fake_tiling);

    fake_tiling.nBL1 = 1;
    fastTillingPtr->isLoad2dFlag = true;
    fastTillingPtr->GetL0TilingRange(fake_tiling);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_01)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    // l0Data_.l0ACurrent / B / C is 9216 / 4608 / 1024;
    // harward_.l0aSize / b / c is 65536 / 32768 / 65536.
    // mTmp / nTmp / kTmp is 2 / 1 / 1
    // L0BScale / A / C is 256 / 512 / 256
    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 8;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 8;
    fake_tiling.nBL1Value = 8;
    // add k,n,m
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_02)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 8;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 8;
    fake_tiling.nBL1Value = 8;

    // build branch case, n,k,m
    fastTilingPtr->hardware_.l1ToL0bRate = 256;
    fastTilingPtr->hardware_.l1ToL0aRate = 512;
    fastTilingPtr->hardware_.l0cToUbRate = 512;
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_03)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 8;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 8;
    fake_tiling.nBL1Value = 8;

    //build branch case, m,k,n
    fastTilingPtr->hardware_.l1ToL0aRate = 512 + 1;
    fastTilingPtr->hardware_.l1ToL0bRate = 256 - 1;
    fastTilingPtr->hardware_.l0cToUbRate = 512;
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_04)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 8;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 8;
    fake_tiling.nBL1Value = 8;

    //build branch case, k,m,n
    fastTilingPtr->hardware_.l1ToL0aRate = 512 + 1;
    fastTilingPtr->hardware_.l1ToL0bRate = 256;
    fastTilingPtr->hardware_.l0cToUbRate = 256;
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_05)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 8;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 8;
    fake_tiling.nBL1Value = 8;

    //build branch case, m,n,k
    fastTilingPtr->hardware_.l1ToL0aRate = 512 + 1;
    fastTilingPtr->hardware_.l1ToL0bRate = 256;
    fastTilingPtr->hardware_.l0cToUbRate = 512 + 2;
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_06)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 8;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 8;
    fake_tiling.nBL1Value = 8;

    //build branch case, n,m,k
    fastTilingPtr->hardware_.l1ToL0aRate = 512 - 1;
    fastTilingPtr->hardware_.l1ToL0bRate = 256;
    fastTilingPtr->hardware_.l0cToUbRate = 512 + 2;
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_special_case)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 8;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 1;
    fake_tiling.nBL1Value = 1;

    // build branch case, n,k,m
    fastTilingPtr->hardware_.l0aSize = 1;
    fastTilingPtr->hardware_.l0bSize = 1;
    fastTilingPtr->hardware_.l0cSize = 1;
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_get_l0_tiling_weight_full_load)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.batchAL1 = 4;
    fake_tiling.groupAL1 = 4;
    fake_tiling.mAL1Value = 4;
    fake_tiling.kAL1ci = 8;
    fake_tiling.kBL1ci = 8;
    fake_tiling.nBL1 = 0;
    fastTilingPtr->hardware_.l0aSize *= 2;
    bool ret = fastTilingPtr->GetL0Tiling(fake_tiling);
    ASSERT_EQ(ret, true);
}

TEST_F(Conv2DFastTilingTest, test_update_ub_data)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    fastTilingPtr->tilingRangeUB_.kAub = {1};
    fastTilingPtr->tilingRangeUB_.mAub = {2};
    fastTilingPtr->tilingRangeUB_.ncFactor = {3};
    fastTilingPtr->tilingRangeUB_.mcFactor = {4};

    fastTilingPtr->opInfo_.preFusionUbUtilize = 5;
    fastTilingPtr->opInfo_.postFusionUbUtilize = 6;

    uint32_t tmpPre = 1 * 2 * 16 * 2 * fastTilingPtr->reduceKAxisAL1_KhDilKwDilCi0_;
    uint32_t preUbCurrent = 5 * tmpPre;
    uint32_t tmpPost = 3 * 4 * 256;
    uint32_t postUbCurrent = 6 * tmpPost;

    fastTilingPtr->UpdateUBData();

    ASSERT_EQ(fastTilingPtr->ubData_.preUbCurrent, preUbCurrent);
    ASSERT_EQ(fastTilingPtr->ubData_.postUbCurrent, postUbCurrent);
}

TEST_F(Conv2DFastTilingTest, test_get_ub_tiling_range)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);
    Tiling fake_tiling;
    fastTilingPtr->GetUBTilingRange(fake_tiling);
}

TEST_F(Conv2DFastTilingTest, test_get_ub_tiling_01)
{
    unique_ptr<FastTiling> fastTilingPtr(new FastTiling());
    fastTilingPtr->SetInputParams(inputParams, hardwareInfo);

    Tiling fake_tiling;
    fake_tiling.kAL1ci = 8;
    fake_tiling.mAL1Value = 8;
    fake_tiling.nC = 8;
    fake_tiling.mC = 1;

    fastTilingPtr->opInfo_.postFusionUbUtilize = hardwareInfo.ubSize / 2 / 256;
    bool ret = fastTilingPtr->GetUBTiling(fake_tiling);
    ASSERT_EQ(ret, true);
}