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

class SEARCH_CACHE_IDX_KERNEL_ST : public testing::Test
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

TEST_F(SEARCH_CACHE_IDX_KERNEL_ST, SearchCacheIdxInt64)
{
    // raw data
    int64_t hashmap[40] = {0, 0, 0, 0,
                           10, 5, -5, 1,
                           2, 1, -5, 1,
                           15, 7, -5, 2,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           3, 3, -5, 1,
                           21, 9, -5, 1};
    int64_t indices[5] = {10, 2, 25, 5, 3};
    int64_t step[1] = {0};
    int64_t emb_max[1] = {25};
    int64_t cache_max[1] = {10};
    int64_t cache_idx[5] = {0};
    int64_t miss_idx[5] = {0};
    int64_t miss_emb_idx[5] = {0};

    int64_t expect_cache_idx[5] = {5, 1, 10, -1, 3};
    int64_t expect_miss_idx[5] = {-1, -1, -1, 3, -1};
    int64_t expect_miss_emb_idx[5] = {-1, -1, -1, 5, -1};

    int64_t expect_hashmap[40] = {0, 0, 0, 0,
                                  10, 5, 0, 1,
                                  2, 1, 0, 1,
                                  15, 7, -5, 2,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  3, 3, 0, 1,
                                  21, 9, -5, 1};

    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "SearchCacheIdx", "SearchCacheIdx")
        .Input({"hashmap", DT_INT64, {10, 4}, (void *)hashmap})
        .Input({"indices", DT_INT64, {5}, (void *)indices})
        .Input({"step", DT_INT64, {1}, (void *)step})
        .Input({"emb_max", DT_INT64, {1}, (void *)emb_max})
        .Input({"cache_max", DT_INT64, {1}, (void *)cache_max})
        .Output({"cache_idx", DT_INT64, {5}, (void *)cache_idx})
        .Output({"miss_idx", DT_INT64, {5}, (void *)miss_idx})
        .Output({"miss_emb_idx", DT_INT64, {5}, (void *)miss_emb_idx});
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(cache_idx, expect_cache_idx, 5 * sizeof(int64_t)));
    EXPECT_EQ(0, std::memcmp(miss_idx, expect_miss_idx, 5 * sizeof(int64_t)));
    EXPECT_EQ(0, std::memcmp(miss_emb_idx, expect_miss_emb_idx, 5 * sizeof(int64_t)));
    EXPECT_EQ(0, std::memcmp(hashmap, expect_hashmap, 40 * sizeof(int64_t)));
}
