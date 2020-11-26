#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include <stdint.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "proto/me_types.pb.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;
using namespace Eigen;

namespace {
    const char* Test  = "RealDiv";
}

class REALDIV_KERNEL_STest : public testing::Test {
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

TEST_F(REALDIV_KERNEL_STest, Host_double)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    double input1[2][2] = {{22, 32},{78,28}};
	//double input2[2][2] = {{2,2},{2,2}};
    double input2[2] = {2,1};
    double output[2][2] = {};
    std::vector<int64_t> v1 = {2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2};
    
	double output_expect[2][2] = {};
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            output_expect[i][j]=input1[i][j]/input2[i];
        }
    }
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_DOUBLE);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(double));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_DOUBLE);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(double));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_DOUBLE);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(double));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
	
	auto outputAddr=outputTensor->GetData();
	double *p=(double *)outputAddr;
    cout << "shuju:" <<endl;
    cout << *p << endl;
    cout << *(p+1) << endl;
    cout << *(p+2) << endl;
    cout << *(p+3) << endl;
    
	EXPECT_EQ(*p, output_expect[0][0]);
	EXPECT_EQ(*(p+1), output_expect[0][1]);
	EXPECT_EQ(*(p+2), output_expect[1][0]);
	EXPECT_EQ(*(p+3), output_expect[1][1]);
}

TEST_F(REALDIV_KERNEL_STest, Host_float)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    float input1[2] = {};
    float input2[2] = {2,1};
    float output[2] = {};
    std::vector<int64_t> v1 = {2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_FLOAT);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(float));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_FLOAT);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(float));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_FLOAT);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(float));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(REALDIV_KERNEL_STest, Host_half)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    half input1[2][2][2] = {};
    half input2[2] = {half(2),half(1)};
    half output[2][2][2] = {};
    std::vector<int64_t> v1 = {2,2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2,2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_FLOAT16);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(half));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_FLOAT16);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(half));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_FLOAT16);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(half));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(REALDIV_KERNEL_STest, Host_uint8_t)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    uint8_t input1[2][2][2][2] = {};
    uint8_t input2[2] = {2,1};
    uint8_t output[2][2][2][2] = {};
    std::vector<int64_t> v1 = {2,2,2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2,2,2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_UINT8);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(uint8_t));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_UINT8);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(uint8_t));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_UINT8);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(uint8_t));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(REALDIV_KERNEL_STest, Host_int8_t)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    int8_t input1[2][2][2][2][2] = {};
    int8_t input2[2] = {2,1};
    int8_t output[2][2][2][2][2] = {};
    std::vector<int64_t> v1 = {2,2,2,2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2,2,2,2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_INT8);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(int8_t));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_INT8);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(int8_t));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_INT8);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(int8_t));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(REALDIV_KERNEL_STest, Host_uint16_t)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    uint16_t input1[2][2][2][2][2][2] = {};
    uint16_t input2[2] = {2,1};
    uint16_t output[2][2][2][2][2][2] = {};
    std::vector<int64_t> v1 = {2,2,2,2,2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2,2,2,2,2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_UINT16);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(uint16_t));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_UINT16);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(uint16_t));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_UINT16);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(uint16_t));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(REALDIV_KERNEL_STest, Host_int16_t)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    int16_t input1[2][2][2][2][2][2][2] = {};
    int16_t input2[2] = {2,1};
    int16_t output[2][2][2][2][2][2][2] = {};
    std::vector<int64_t> v1 = {2,2,2,2,2,2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2,2,2,2,2,2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_INT16);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(int16_t));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_INT16);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(int16_t));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_INT16);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(int16_t));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(REALDIV_KERNEL_STest, Host_int32_t)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    int32_t input1[2][2][2] = {};
    int32_t input2[2] = {2,1};
    int32_t output[2][2][2] = {};
    std::vector<int64_t> v1 = {2,2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2,2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_INT32);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(int32_t));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_INT32);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(int32_t));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_INT32);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(int32_t));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(REALDIV_KERNEL_STest, Host_int64_t)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("RealDiv");

    int64_t input1[2][2][2] = {};
    int64_t input2[2] = {2,1};
    int64_t output[2][2][2] = {};
    std::vector<int64_t> v1 = {2,2,2};
    std::vector<int64_t> v2 = {2};
    std::vector<int64_t> v3 = {2,2,2};
    
//input1
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);

    inputTensor1->SetDataType(aicpu::DT_INT64);
    inputTensor1->SetData(input1);
    //inputTensor1->SetDataSize(4 * sizeof(int64_t));
	
//input2
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);

    inputTensor2->SetDataType(aicpu::DT_INT64);
    inputTensor2->SetData(input2);
    //inputTensor2->SetDataSize(4 * sizeof(int64_t));
//output
	auto outputTensor = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor, nullptr);

    outputTensor->SetDataType(aicpu::DT_INT64);
    outputTensor->SetData(output);
    //outputTensor->SetDataSize(4 * sizeof(int64_t));

    auto Shape1 = inputTensor1->GetTensorShape();
    auto Shape2 = inputTensor2->GetTensorShape();
    auto Shape3 = outputTensor->GetTensorShape();
    Shape1->SetDimSizes(v1);
    Shape2->SetDimSizes(v2);
    Shape3->SetDimSizes(v3);

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

