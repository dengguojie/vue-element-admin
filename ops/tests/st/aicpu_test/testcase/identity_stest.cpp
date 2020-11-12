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

class IDENTITY_KERNEL_ST : public testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }

 private:
};

TEST_F(IDENTITY_KERNEL_ST, IdentityInt64) {
    // raw data
    int64_t input[4] = {1,2,3,4};
    int64_t out[4] = {0};
    int64_t expect_out[4] = {1,2,3,4};
    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Identity", "Identity")
        .Input({"x", DT_INT64, {2, 2}, (void *)input})
        .Output({"y", DT_INT64, {2, 2}, (void *)out});
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(out, expect_out, 4 * sizeof(int64_t)));
}
