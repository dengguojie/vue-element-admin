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

class MESHGRID_KERNEL_ST : public testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }

 private:
};

TEST_F(MESHGRID_KERNEL_ST, MeshgridInt64) {
    // raw data
    int input0[4] = {1, 2, 3, 1};
    int input1[3] = {4, 5, 6};
    int input2[6] = {7, 8, 9, 4, 2, 2};

    int output1[72] = {0};
    int output2[72] = {0};
    int output3[72] = {0};

    int expected1[72] = {1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3,
                         1, 1, 1, 1, 1, 1};
    int expected2[72] = {4, 4, 4, 4, 4, 4,
                         4, 4, 4, 4, 4, 4,
                         4, 4, 4, 4, 4, 4,
                         4, 4, 4, 4, 4, 4,
                         5, 5, 5, 5, 5, 5,
                         5, 5, 5, 5, 5, 5,
                         5, 5, 5, 5, 5, 5,
                         5, 5, 5, 5, 5, 5,
                         6, 6, 6, 6, 6, 6,
                         6, 6, 6, 6, 6, 6,
                         6, 6, 6, 6, 6, 6,
                         6, 6, 6, 6, 6, 6};
    int expected3[72] = {7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2};
    std::string indexing = "xy";

    // nodeDef
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Meshgrid", "Meshgrid")
        .Input({"input0", DT_INT32, {4}, (void *)input0})
        .Input({"input1", DT_INT32, {3}, (void *)input1})
        .Input({"input2", DT_INT32, {6}, (void *)input2})
        .Output({"output1", DT_INT32, {3, 4, 6}, (void *)output1})
        .Output({"output2", DT_INT32, {3, 4, 6}, (void *)output2})
        .Output({"output3", DT_INT32, {3, 4, 6}, (void *)output3})
        .Attr("indexing", indexing);
    // excute
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(output1, expected1, 72 * sizeof(int)));
    EXPECT_EQ(0, std::memcmp(output2, expected2, 72 * sizeof(int)));
    EXPECT_EQ(0, std::memcmp(output3, expected3, 72 * sizeof(int)));
}
