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
#include "proto/me_types.pb.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include <mockcpp/ChainingMockHelper.h>

using namespace std;
using namespace aicpu;


class TEST_Less_UTest : public testing::Test {
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

TEST_F(TEST_Less_UTest, Int32Nor1)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    int32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(int32_t));

    //set x2
    int32_t x2[ind] = {1, 2, 3,
                           14, 5, 13,
                           7, 18, 3,
                           1,1,2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(int32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, Int64Nor1)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    int64_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT64);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(int64_t));

    //set x2
    int64_t x2[ind] = {1, 2, 3,
                           14, 5, 13,
                           7, 18, 3,
                           1,1,2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_INT64);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(int64_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, FloatNor1)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    float x1[ind] = {1, 2, 3.5,
                           4, 5.8, 3,
                           7, 8, 3,
                           11,12.7,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_FLOAT);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(float));

    //set x2
    float x2[ind] = {1, 2, 3.8,
                           14, 5.9, 13,
                           7, 18, 3,
                           1,1.6,2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_FLOAT);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(float));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, DoubleNor1)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    double x1[ind] = {1, 2, 3.5,
                           4, 5.8, 3,
                           7, 8, 3,
                           11,12.7,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {1,1,2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_DOUBLE);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(double));

    //set x2
    double x2[ind] = {1, 2, 3.8,
                           14, 5.9, 13,
                           7, 18, 3,
                           1,1.6,2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_DOUBLE);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(double));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, Int32Nor2)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    int32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {1,2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(int32_t));

    //set x2
    int32_t x2[ind] = {1, 2, 3,
                           14, 5, 13,
                           7, 18, 3,
                           1,1,2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(int32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, Int32Nor3)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 18;
    //set x1
    int32_t x1[ind] = {1, 2, 3, 4, 5, 3, 7, 8, 3, 11,12,23,11,12,23,11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,9};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(int32_t));

    //set x2
    int32_t x2[ind] = {1, 2, 3,
                           14, 5, 13,
                           7, 18, 3,
                           1,1,2,1, 2, 3, 4, 5, 3};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(int32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, Int32Nor4)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 1;
    //set x1
    int32_t x1[ind] = {1};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {1};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(int32_t));

    //set x2
    int32_t x2[ind] = {2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(int32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, UInt32Nor2)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    uint32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_UINT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(uint32_t));

    //set x2
    uint32_t x2[ind] = {1, 2, 3,
                           14, 5, 13,
                           7, 18, 3,
                           1,1,2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_UINT32);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(uint32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, Int32BroadcastNor)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    uint32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(uint32_t));

    //set x2
    uint32_t x2 = 7;

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    std::vector<int64_t> shape2 = {1,1};
    x2Shape->SetDimSizes(shape2);
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(&x2);
    x2Tensor->SetDataSize(ind * sizeof(uint32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(uint32_t));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_Less_UTest, ExpDiffShape3)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    uint32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(uint32_t));

    //set x2
    uint32_t x2[4] = {7,4,5,6};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    std::vector<int64_t> shape2 = {2,2};
    x2Shape->SetDimSizes(shape2);
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(4 * sizeof(uint32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(uint32_t));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Less_UTest, ExpInput4)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    uint32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(uint32_t));

    //set x2
    uint32_t x2[16] = {7,4,5,6,7,4,5,6,7,4,5,6,7,4,5,6};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    std::vector<int64_t> shape2 = {2,2,4};
    x2Shape->SetDimSizes(shape2);
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(16 * sizeof(uint32_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(uint32_t));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Less_UTest, ExpInputDiffDtype5)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    uint32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(uint32_t));

    //set x2
    uint64_t x2[ind] = {1, 2, 3,
                           14, 5, 13,
                           7, 18, 3,
                           1,1,2};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_INT64);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(uint64_t));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Less_UTest, ExpInputNull6)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    uint32_t x1[ind] = {1, 2, 3,
                           4, 5, 3,
                           7, 8, 3,
                           11,12,23};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,2,3};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(uint32_t));

    //set x2
    uint64_t x2[ind] = {1, 2, 3,
                           14, 5, 13,
                           7, 18, 3,
                           1,1,2};

    auto x2Tensor = nodeDef->AddInputs();
    x2Tensor = nullptr;


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Less_UTest, ExpInputNull7)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    auto x1Tensor = nodeDef->AddInputs();
    x1Tensor = nullptr;
    
    auto x2Tensor = nodeDef->AddInputs();
    x2Tensor = nullptr;


    //set output
    bool output[1] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(1 * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Less_UTest, ExpIntputBool8)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Less");

    const int32_t ind = 12;
    //set x1
    bool x1[ind] = {1};

    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);

    auto x1Shape = x1Tensor->GetTensorShape();
    std::vector<int64_t> shapes = {2,6};
    x1Shape->SetDimSizes(shapes);
    x1Tensor->SetDataType(DT_BOOL);
    x1Tensor->SetData(x1);
    x1Tensor->SetDataSize(ind * sizeof(bool));

    //set x2
    bool x2[ind] = {0};

    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);

    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes(shapes);
    x2Tensor->SetDataType(DT_BOOL);
    x2Tensor->SetData(x2);
    x2Tensor->SetDataSize(ind * sizeof(bool));


    //set output
    bool output[ind] = {0};
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    auto yShape = yTensor->GetTensorShape();
    yShape->SetDimSizes(shapes);
    yTensor->SetDataType(DT_BOOL);
    yTensor->SetData(output);
    yTensor->SetDataSize(ind * sizeof(bool));


    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}