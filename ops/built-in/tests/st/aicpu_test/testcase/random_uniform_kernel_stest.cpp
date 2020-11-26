#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace aicpu;

class TEST_RANDOM_UNIFORM_KERNEL_STest : public testing::Test {
  protected:
    virtual void SetUp() {}

    virtual void TearDown() {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_RANDOM_UNIFORM_KERNEL_STest, float16_with_seed) {
    cout<<"Test Kernel Begin."<<endl;
    int64_t input[2] = {2, 2};
    Eigen::half output[4] = {static_cast<Eigen::half>(0)};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "RandomUniform", "RandomUniform")
        .Input({"shape", DT_INT64, {2}, (void *)input})
        .Output({"y", DT_FLOAT16, {2, 2}, (void *)output})
        .Attr("seed", 10)
        .Attr("dtype", DT_FLOAT16);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_NE(output[0], output[1]);
    EXPECT_NE(output[0], output[2]);
    EXPECT_NE(output[0], output[3]);
    EXPECT_NE(output[1], output[2]);
    EXPECT_NE(output[1], output[3]);
    EXPECT_NE(output[2], output[3]);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_RANDOM_UNIFORM_KERNEL_STest, float_with_seed_2) {
    cout<<"Test Kernel Begin."<<endl;
    int64_t input[2] = {2, 2};
    float output[4] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "RandomUniform", "RandomUniform")
        .Input({"shape", DT_INT64, {2}, (void *)input})
        .Output({"y", DT_FLOAT, {2, 2}, (void *)output})
        .Attr("seed", 0)
        .Attr("seed2", 10)
        .Attr("dtype", DT_FLOAT);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_NE(output[0], output[1]);
    EXPECT_NE(output[0], output[2]);
    EXPECT_NE(output[0], output[3]);
    EXPECT_NE(output[1], output[2]);
    EXPECT_NE(output[1], output[3]);
    EXPECT_NE(output[2], output[3]);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_RANDOM_UNIFORM_KERNEL_STest, double_with_no_seed) {
    cout<<"Test Kernel Begin."<<endl;
    int64_t input[2] = {2, 2};
    double output[4] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "RandomUniform", "RandomUniform")
        .Input({"shape", DT_INT64, {2}, (void *)input})
        .Output({"y", DT_DOUBLE, {2, 2}, (void *)output})
        .Attr("dtype", DT_DOUBLE);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_NE(output[0], output[1]);
    EXPECT_NE(output[0], output[2]);
    EXPECT_NE(output[0], output[3]);
    EXPECT_NE(output[1], output[2]);
    EXPECT_NE(output[1], output[3]);
    EXPECT_NE(output[2], output[3]);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_RANDOM_UNIFORM_KERNEL_STest, data_type_not_match) {
    cout<<"Test Kernel Begin."<<endl;
    int64_t input[2] = {2, 2};
    double output[4] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "RandomUniform", "RandomUniform")
        .Input({"shape", DT_INT64, {2}, (void *)input})
        .Output({"y", DT_DOUBLE, {2, 2}, (void *)output})
        .Attr("dtype", DT_INT64);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_NE(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 0);
    EXPECT_EQ(output[3], 0);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_RANDOM_UNIFORM_KERNEL_STest, data_type_not_support) {
    cout<<"Test Kernel Begin."<<endl;
    int64_t input[2] = {2, 2};
    int64_t output[4] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "RandomUniform", "RandomUniform")
        .Input({"shape", DT_INT64, {2}, (void *)input})
        .Output({"y", DT_INT64, {2, 2}, (void *)output})
        .Attr("dtype", DT_INT64);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_NE(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 0);
    EXPECT_EQ(output[3], 0);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}