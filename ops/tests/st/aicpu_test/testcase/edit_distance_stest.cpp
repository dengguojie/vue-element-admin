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

class EDIT_DISTANCE_KERNEL_ST : public testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }

 private:
};

TEST_F(EDIT_DISTANCE_KERNEL_ST, EditDistanceInt64) {
    // raw data
    int64_t hi[9] = {0,0,0,1,0,1,1,1,1};
    int64_t hv[3] = {1,2,3};
    int64_t hs[3] = {2,2,2};
    int64_t ti[12] = {0,1,0,0,0,1,1,1,0,1,0,1};
    int64_t tv[4] = {1,2,3,1};
    int64_t ts[3] = {2,2,2};
    float out[4] = {0};
    float expect_out[4] = {1, 1, 1, 0};
    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "EditDistance", "EditDistance")
        .Input({"hypothesis_indices", DT_INT64, {3, 3}, (void *)hi})
        .Input({"hypothesis_values", DT_INT64, {3}, (void *)hv})
        .Input({"hypothesis_shape", DT_INT64, {3}, (void *)hs})
        .Input({"truth_indices", DT_INT64, {4, 3}, (void *)ti})
        .Input({"truth_values", DT_INT64, {4}, (void *)tv})
        .Input({"truth_shape", DT_INT64, {3}, (void *)ts})
        .Output({"output", DT_FLOAT, {2, 2}, (void *)out})
        .Attr("normalize", true);
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(out, expect_out, 4 * sizeof(float)));
}
