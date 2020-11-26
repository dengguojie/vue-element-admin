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

#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "proto/me_types.pb.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

using namespace std;
using namespace aicpu;
using namespace Eigen;

namespace {
    const char* Test  = "Ceil";
}

class CEIL_KERNEL_UTest : public testing::Test {
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

TEST_F(CEIL_KERNEL_UTest, Host_double)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    double input[2][2] = {{22.0, 32.3},{-78.0,-28.5}};
    double output[2][2] = {};
	double output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=ceil(input[i][j]);
        }
    }

    NodeDefBuilder(nodeDef.get(), "Ceil", "Ceil")
        .Input({"x", DT_DOUBLE, {4}, input})
        .Output({"y", DT_DOUBLE, {4}, output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    double *p=(double *)output;
    EXPECT_EQ(*p, output_expect[0][0]);
    EXPECT_EQ(*(p+1), output_expect[0][1]);
    EXPECT_EQ(*(p+2), output_expect[1][0]);
    EXPECT_EQ(*(p+3), output_expect[1][1]);

}

TEST_F(CEIL_KERNEL_UTest, Host_float)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    float input[2][2] = {{22.0, 32.3},{-78.0,-28.5}};
    float output[2][2] = {};
	float output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=ceil(input[i][j]);
        }
    }

    NodeDefBuilder(nodeDef.get(), "Ceil", "Ceil")
        .Input({"x", DT_FLOAT, {4}, input})
        .Output({"y", DT_FLOAT, {4}, output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    float *p=(float *)output;
    EXPECT_EQ(*p, output_expect[0][0]);
    EXPECT_EQ(*(p+1), output_expect[0][1]);
    EXPECT_EQ(*(p+2), output_expect[1][0]);
    EXPECT_EQ(*(p+3), output_expect[1][1]);
}

TEST_F(CEIL_KERNEL_UTest, Host_half)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    half input[2][2] = {{half(12.0), half(19.3)},{half(-8.0),half(-8.5)}};
    half output[2][2] = {};
	half output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=ceil(input[i][j]);
        }
    }

    NodeDefBuilder(nodeDef.get(), "Ceil", "Ceil")
        .Input({"x", DT_FLOAT16, {4}, input})
        .Output({"y", DT_FLOAT16, {4}, output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    half *p=(half *)output;
    EXPECT_EQ(*p, output_expect[0][0]);
    EXPECT_EQ(*(p+1), output_expect[0][1]);
    EXPECT_EQ(*(p+2), output_expect[1][0]);
    EXPECT_EQ(*(p+3), output_expect[1][1]);
}

TEST_F(CEIL_KERNEL_UTest, Host_int32_t)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    int32_t input[2][2] = {{22.0, 32.3},{-78.0,-28.5}};
    int32_t output[2][2] = {};
	int32_t output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=ceil(input[i][j]);
        }
    }

    NodeDefBuilder(nodeDef.get(), "Ceil", "Ceil")
        .Input({"x", DT_INT32, {4}, input})
        .Output({"y", DT_INT32, {4}, output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}
