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
#include "cpu_types.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;
using namespace Eigen;

namespace {
    const char* Test  = "Ceil";
}

class CEIL_KERNEL_STest : public testing::Test {
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

TEST_F(CEIL_KERNEL_STest, Host_double)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Ceil");

    double input[2][2] = {{22.0, 32.3},{-78.0,-28.5}};
    double output[2][2] = {};
	double output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=ceil(input[i][j]);
        }
    }
//input
    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);

    inputTensor->SetDataType((int32_t)aicpu::DT_DOUBLE);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(4 * sizeof(double));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType((int32_t)aicpu::DT_DOUBLE);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(4 * sizeof(double));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
	
	auto outputAddr=outputTensor->GetData();
	double *p=(double *)outputAddr;
	EXPECT_EQ(*p, output_expect[0][0]);
	EXPECT_EQ(*(p+1), output_expect[0][1]);
	EXPECT_EQ(*(p+2), output_expect[1][0]);
	EXPECT_EQ(*(p+3), output_expect[1][1]);

}

TEST_F(CEIL_KERNEL_STest, Host_float)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Ceil");

    float input[2][2] = {{22.0, 32.3},{-78.0,-28.5}};
    float output[2][2] = {};
	float output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=ceil(input[i][j]);
        }
    }
//input
    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);

    inputTensor->SetDataType((int32_t)aicpu::DT_FLOAT);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(4 * sizeof(float));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType((int32_t)aicpu::DT_FLOAT);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(4 * sizeof(float));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
	
	auto outputAddr=outputTensor->GetData();
	float *p=(float *)outputAddr;
	EXPECT_EQ(*p, output_expect[0][0]);
	EXPECT_EQ(*(p+1), output_expect[0][1]);
	EXPECT_EQ(*(p+2), output_expect[1][0]);
	EXPECT_EQ(*(p+3), output_expect[1][1]);

}

TEST_F(CEIL_KERNEL_STest, Host_half)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Ceil");

    half input[2][2] = {{half(12.0), half(19.3)},{half(-8.0),half(-8.5)}};
    half output[2][2] = {};
	half output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=ceil(input[i][j]);
        }
    }
//input
    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);

    inputTensor->SetDataType((int32_t)aicpu::DT_FLOAT16);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(4 * sizeof(half));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType((int32_t)aicpu::DT_FLOAT16);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(4 * sizeof(half));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
	
	auto outputAddr=outputTensor->GetData();
	half *p=(half *)outputAddr;
	EXPECT_EQ(*p, output_expect[0][0]);
	EXPECT_EQ(*(p+1), output_expect[0][1]);
	EXPECT_EQ(*(p+2), output_expect[1][0]);
	EXPECT_EQ(*(p+3), output_expect[1][1]);

}
