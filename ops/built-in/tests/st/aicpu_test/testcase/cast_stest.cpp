#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include <math.h>
#include <stdint.h>
#include <Eigen/Dense>

#include "node_def_builder.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;
using namespace Eigen;

namespace {
const char *Test = "Cast";
}

class CAST_KERNEL_STest : public testing::Test {
protected:
    virtual void SetUp() {}

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

TEST_F(CAST_KERNEL_STest, Host_FLOAT_INT)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Cast");

    float input[22] = {22, 32.3, -78.0, -28.5, 77, 99, 77, 89, 22, 32.3, -78.0,
        -28.5, 77, 99, 77, 45.7, 89.5, 90, 2, 1, 22, 32.3};
    int output[22] = {0};
    int expect_out[2][2] = {22, 32, -78, -28};
    NodeDefBuilder(nodeDef.get(), "Cast", "Cast")
        .Input({ "x", DT_FLOAT, { 2, 11 }, (void *)input })
        .Output({ "y", DT_INT32, { 2, 11 }, (void *)output });

    CpuKernelContext ctx(HOST);
    int length = sizeof(input) / sizeof(float);
    cout << "length" << length << endl;
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    // EXPECT_EQ(0, std::memcmp(output, expect_out, 4 * sizeof(int32_t)));
    EXPECT_EQ(output[0], 22);
    EXPECT_EQ(output[1], 32);
    EXPECT_EQ(output[20], 22);
    EXPECT_EQ(output[21], 32);
    cout << "Test Kernel " << nodeDef->GetOpType() << " Finish. " << endl;
}
TEST_F(CAST_KERNEL_STest, Host_FLOAT_INT1)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Cast");

    float input[20] = {22, 32.3, -78.0, -28.5, 77, 99, 77, 89, 22, 32.3, 22, 32.3 , -78.0, -28.5, 77, 99, 77, 45.7, 89.5, 90};//, 2, 1
    int output[10] = {0};
    int expect_out[2][2] = {22, 32, -78, -28};
    NodeDefBuilder(nodeDef.get(), "Cast", "Cast")
        .Input({ "x", DT_FLOAT, { 2, 5 ,2}, (void *)input })
        .Output({ "y", DT_INT32, { 2, 5 }, (void *)output });

    CpuKernelContext ctx(HOST);
    int length = sizeof(input) / sizeof(float);
    cout << "length" << length << endl;
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    // EXPECT_EQ(0, std::memcmp(output, expect_out, 4 * sizeof(int32_t)));
    EXPECT_EQ(output[0], 22);
    EXPECT_EQ(output[1], 32);
    cout << "Test Kernel " << nodeDef->GetOpType() << " Finish. " << endl;
}
TEST_F(CAST_KERNEL_STest, Host_FLOAT_BOOL)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Cast");

    float input[22] = {0, 32.3, -78.0, -28.5, 77, 99, 77, 89, 22, 32.3, -78.0,
        -28.5, 77, 99, 77, 45.7, 89.5, -90, 2, 1, 0, -32.3};
    bool output[22] = {true};
    int expect_out[2][2] = {22,32, -78, -28};
    NodeDefBuilder(nodeDef.get(), "Cast", "Cast")
        .Input({ "x", DT_FLOAT, { 2, 11 }, (void *)input })
        .Output({ "y", DT_BOOL, { 2, 11 }, (void *)output });

    CpuKernelContext ctx(HOST);
    int length = sizeof(input) / sizeof(float);
    cout << "length" << length << endl;
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    // EXPECT_EQ(0, std::memcmp(output, expect_out, 4 * sizeof(int32_t)));
    EXPECT_EQ(output[0], false);
    EXPECT_EQ(output[1], true);
    EXPECT_EQ(output[20], false);
    EXPECT_EQ(output[21], true);
    cout << "Test Kernel " << nodeDef->GetOpType() << " Finish. " << endl;
}

TEST_F(CAST_KERNEL_STest, Host_INT_BOOL)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Cast");

    int input[22] = {0, 32, -78, -28, 77, 99 ,77, 89, 22, 32.3, -78, -28,77, 99 ,77 ,45, 89, -90, 2, 1, 0, -32};
    bool output[22] = {true};
    int expect_out[2][2] = {22,32, -78, -28};
    NodeDefBuilder(nodeDef.get(), "Cast", "Cast")
        .Input({ "x", DT_FLOAT, { 2, 11 }, (void *)input })
        .Output({ "y", DT_BOOL, { 2, 11 }, (void *)output });

    CpuKernelContext ctx(HOST);
    int length = sizeof(input) / sizeof(float);
    cout << "length" << length << endl;
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    // EXPECT_EQ(0, std::memcmp(output, expect_out, 4 * sizeof(int32_t)));
    EXPECT_EQ(output[0], false);
    EXPECT_EQ(output[1], true);
    EXPECT_EQ(output[20], false);
    EXPECT_EQ(output[21], true);
    cout << "Test Kernel " << nodeDef->GetOpType() << " Finish. " << endl;
}