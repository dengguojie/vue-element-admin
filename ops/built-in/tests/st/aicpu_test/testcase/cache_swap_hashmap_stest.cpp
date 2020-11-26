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

class CACHE_SWAP_HASHMAP_KERNEL_ST : public testing::Test
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

TEST_F(CACHE_SWAP_HASHMAP_KERNEL_ST, CacheSwapHashmapInt64)
{
    // raw data for indices : {10, 2, 20, 5, 3}
    int64_t hashmap[40] = {0, 0, 0, 0,
                           10, 5, 0, 1,
                           2, 1, 0, 1,
                           15, 7, -5, 2,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           3, 3, 0, 1,
                           21, 9, -5, 1};
    int64_t miss_emb_idx[5] = {-1, -1, 20, 5, -1};
    int64_t step[1] = {0};
    int64_t swap_cache_idx[5] = {0};
    int64_t old_emb_idx[5] = {0};

    int64_t expect_swap_cache_idx[5] = {-1, -1, 9, 7, -1};
    int64_t expect_old_emb_idx[5] = {-1, -1, 21, 15, -1};

    int64_t expect_hashmap[40] = {5, 7, 0, 1,
                                  10, 5, 0, 1,
                                  2, 1, 0, 1,
                                  20, 9, 0, 1,
                                  20, 9, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  3, 3, 0, 1,
                                  21, 9, -5, 0};

    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "CacheSwapHashmap", "CacheSwapHashmap")
        .Input({"hashmap", DT_INT64, {10, 4}, (void *)hashmap})
        .Input({"miss_emb_idx", DT_INT64, {5}, (void *)miss_emb_idx})
        .Input({"step", DT_INT64, {1}, (void *)step})
        .Output({"swap_cache_idx", DT_INT64, {5}, (void *)swap_cache_idx})
        .Output({"old_emb_idx", DT_INT64, {5}, (void *)old_emb_idx});
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(swap_cache_idx, expect_swap_cache_idx, 5 * sizeof(int64_t)));
    EXPECT_EQ(0, std::memcmp(old_emb_idx, expect_old_emb_idx, 5 * sizeof(int64_t)));
    EXPECT_EQ(0, std::memcmp(hashmap, expect_hashmap, 40 * sizeof(int64_t)));
}
