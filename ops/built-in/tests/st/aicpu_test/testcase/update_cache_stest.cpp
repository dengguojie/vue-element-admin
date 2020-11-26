#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "node_def_builder.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;

namespace
{
    const char *Test = "Test";
}

struct InputOutputNode1
{
    std::string node;
    aicpu::DataType dType;
};

class UPDATE_CACHE_KERNEL_ST : public testing::Test
{
protected:
    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

TEST_F(UPDATE_CACHE_KERNEL_ST, UpdateCacheFloat32)
{
    // raw data
    float cache_table[40] = {0, 0, 0, 0,
                             10, 5, 0, 1,
                             2, 1, 0, 1,
                             15, 7, -5, 2,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             3, 3, 0, 1,
                             21, 9, -5, 1};
    int64_t indices[5] = {-1, 50, 2, 3, -1};
    float update[20] = {0, 0, 0, 0,
                        55, 55, 55, 55,
                        22, 22, 22, 22,
                        33, 33, 33, 33,
                        11, 11, 11, 11};
    int64_t max_num[1] = {50};
    float out[1] = {0};

    float expect_out[1] = {0};

    float expect_cache_table[40] = {0, 0, 0, 0,
                                    10, 5, 0, 1,
                                    22, 22, 22, 22,
                                    33, 33, 33, 33,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    3, 3, 0, 1,
                                    21, 9, -5, 1};

    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "UpdateCache", "UpdateCache")
        .Input({"cache_table", DT_FLOAT, {10, 4}, (void *)cache_table})
        .Input({"indices", DT_INT64, {5}, (void *)indices})
        .Input({"update", DT_FLOAT, {5, 4}, (void *)update})
        .Input({"max_num", DT_INT64, {1}, (void *)max_num})
        .Output({"out", DT_FLOAT, {1}, (void *)out});
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(out, expect_out, 1 * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(cache_table, expect_cache_table, 40 * sizeof(float)));
}
