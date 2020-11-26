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

namespace {
const char* Test  = "Test";
}

struct InputOutputNode1{
    std::string node;
    aicpu::DataType dType;
};

class GATHER_KERNEL_ST : public testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }

 private:
};

TEST_F(GATHER_KERNEL_ST, GatherInt64) {
    // raw data
    int64_t input[4] = {1,2,3,4};
    int64_t dim[1] = {1};
    int64_t index[4] = {0,0,1,0};
    int64_t out[4] = {0};
    int64_t expect_out[4] = {1, 1, 4, 3};
    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "GatherD", "GatherD")
        .Input({"input", DT_INT64, {2, 2}, (void *)input})
        .Input({"dim", DT_INT64, {1}, (void *)dim})
        .Input({"index", DT_INT64, {2, 2}, (void *)index})
        .Output({"output", DT_INT64, {2, 2}, (void *)out});
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(out, expect_out, 4 * sizeof(int64_t)));
}

TEST_F(GATHER_KERNEL_ST, GatherInt64_2) {
    // raw data
    int64_t input[9] = {1,2,8,3,4,9,7,6,5};
    int dim[1] = {0};
    int64_t index[6] = {0,0,2,1,0,2};
    int64_t out[6] = {0};
    int64_t expect_out[6] = {1,2,5,3,2,5};
    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "GatherD", "GatherD")
        .Input({"input", DT_INT64, {3, 3}, (void *)input})
        .Input({"dim", DT_INT32, {1}, (void *)dim})
        .Input({"index", DT_INT64, {2, 3}, (void *)index})
        .Output({"output", DT_INT64, {2, 3}, (void *)out});
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(out, expect_out, 6 * sizeof(int64_t)));
}
