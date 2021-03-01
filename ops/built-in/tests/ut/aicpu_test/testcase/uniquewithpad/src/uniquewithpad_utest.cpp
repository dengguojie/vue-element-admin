#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include <mockcpp/ChainingMockHelper.h>

using namespace std;
using namespace aicpu;


class TEST_Unique_UTest : public testing::Test {
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

TEST_F(TEST_Unique_UTest, Int32Nor1)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("UniqueWithPad");

    const int32_t ind = 9;
    //set x
    uint32_t input[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3};

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);

    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {1,3,3};
    aicpuShape->SetDimSizes(shapes);
    inputTensor->SetDataType(DT_INT32);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(ind * sizeof(uint32_t));

    //set k
    uint32_t padding = 21;
    auto input_padding_ = nodeDef->AddInputs();
    EXPECT_NE(input_padding_, nullptr);
    input_padding_->SetDataType(DT_INT32);
    input_padding_->SetData(&padding);


    //set output
    uint32_t output[ind] = {0};
    uint32_t indices[ind] = {0};
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint32_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(uint32_t));

    // auto sorted = CreateAttrValue();
    // sorted->SetBool(true);
    // nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Unique_UTest, Int64Nor2)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("UniqueWithPad");

    const int32_t ind = 9;
    //set x
    uint64_t input[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3};

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {1,3,3};
    aicpuShape->SetDimSizes(shapes);
    inputTensor->SetDataType(DT_INT64);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(ind * sizeof(uint64_t));

    //set k
    uint64_t padding = 21;
    auto input_padding_ = nodeDef->AddInputs();
    EXPECT_NE(input_padding_, nullptr);
    input_padding_->SetDataType(DT_INT64);
    input_padding_->SetData(&padding);

    uint64_t output[ind] = {0};
    uint64_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT64);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint64_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT64);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(uint64_t));

    // auto sorted = CreateAttrValue();
    // sorted->SetBool(true);
    // nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Unique_UTest, ExpDType3)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("UniqueWithPad");

    const int32_t ind = 9;
    //set x
    uint32_t input[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3};

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {1,3,3};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_FLOAT);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(ind * sizeof(float));

    //set padding
    uint32_t padding = 21;
    auto input_padding_ = nodeDef->AddInputs();
    EXPECT_NE(input_padding_, nullptr);
    input_padding_->SetDataType(DT_INT32);
    input_padding_->SetData(&padding);

    uint32_t output[ind] = {0};
    uint32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint32_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(uint32_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Unique_UTest, ExpInput4)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("UniqueWithPad");

    const int32_t ind = 9;
    //set x
    uint32_t input[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3};

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {1,3,3};
    aicpuShape->SetDimSizes(shapes);
    inputTensor->SetDataType(DT_INT32);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(ind * sizeof(uint32_t));

    //set output
    uint32_t output[ind] = {0};
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint32_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Unique_UTest, ExpOutput5)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("UniqueWithPad");

    const int32_t ind = 9;
    //set x
    uint64_t input[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3};

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {1,3,3};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_INT64);

    inputTensor->SetData(input);
    inputTensor->SetDataSize(ind * sizeof(uint64_t));

    //set padding
    uint32_t padding = 21;
    auto input_padding_ = nodeDef->AddInputs();
    EXPECT_NE(input_padding_, nullptr);
    input_padding_->SetDataType(DT_INT32);
    input_padding_->SetData(&padding);

    //set output
    uint32_t output[ind] = {0};
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint64_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}