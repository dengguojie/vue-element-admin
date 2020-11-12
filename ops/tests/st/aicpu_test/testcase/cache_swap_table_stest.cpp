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

class CACHE_SWAP_TABLE_KERNEL_ST : public testing::Test
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

TEST_F(CACHE_SWAP_TABLE_KERNEL_ST, CacheSwapTableFloat32)
{
    // raw data for indices : {10, 2, 20, 5, 3}
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
    int64_t swap_cache_idx[5] = {-1, -1, 2, 3, -1};
    float miss_value[20] = {0, 0, 0, 0,
                            0, 0, 0, 0,
                            22, 22, 22, 22,
                            33, 33, 33, 33,
                            11, 11, 11, 11};
    float old_value[20] = {0};

    float expect_old_value[20] = {0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  2, 1, 0, 1,
                                  15, 7, -5, 2,
                                  0, 0, 0, 0};

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
    NodeDefBuilder(nodeDef.get(), "CacheSwapTable", "CacheSwapTable")
        .Input({"cache_table", DT_FLOAT, {10, 4}, (void *)cache_table})
        .Input({"swap_cache_idx", DT_INT64, {5}, (void *)swap_cache_idx})
        .Input({"miss_value", DT_FLOAT, {5, 4}, (void *)miss_value})
        .Output({"old_value", DT_FLOAT, {5, 4}, (void *)old_value});
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(old_value, expect_old_value, 20 * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(cache_table, expect_cache_table, 40 * sizeof(float)));
}
