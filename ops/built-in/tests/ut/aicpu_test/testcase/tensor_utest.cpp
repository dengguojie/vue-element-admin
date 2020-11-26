#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "cpu_tensor.h"
#include "device.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;

class TENSOR_UTest : public testing::Test {
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

TEST_F(TENSOR_UTest, TensorShape)
{
    auto shape = CpuKernelUtils::CreateTensorShape();
    shape->SetFormat(FORMAT_NCHW);
    vector<int64_t> dims = {3,1,3,5};
    shape->SetDimSizes(dims);
    shape->SetUnknownRank(false);
    auto tensor = CpuKernelUtils::CreateTensor();
    tensor->SetTensorShape(shape.get());

    int64_t size = tensor->NumElements();
    EXPECT_EQ(size, 3 * 1 * 5 * 3);

    auto outShape = tensor->GetTensorShape();
    int32_t dimSize = outShape->GetDims();
    EXPECT_EQ(dimSize, 4);
    EXPECT_EQ(outShape->GetDimSize(0), 3);
    EXPECT_EQ(outShape->GetDimSize(1), 1);
    EXPECT_EQ(outShape->GetDimSize(2), 3);
    EXPECT_EQ(outShape->GetDimSize(3), 5);

    EXPECT_EQ(outShape->GetUnknownRank(), false);

    vector<int64_t> retDims = outShape->GetDimSizes();
    EXPECT_EQ(retDims.size(), 4);
    EXPECT_EQ(retDims[0], 3);
    EXPECT_EQ(retDims[1], 1);
    EXPECT_EQ(retDims[2], 3);
    EXPECT_EQ(retDims[3], 5);

    int32_t format = shape->GetFormat();
    EXPECT_EQ(format, FORMAT_NCHW);
}

TEST_F(TENSOR_UTest, CalcDataSizeByShapeFailed)
{
    MOCKER(GetSizeByDataType)
    .stubs()
    .will(returnValue(-1));

    auto tensor = CpuKernelUtils::CreateTensor();
    int64_t size = tensor->CalcDataSizeByShape();
    EXPECT_LT(size, 0);
}

TEST_F(TENSOR_UTest, DataType)
{
    auto tensor = CpuKernelUtils::CreateTensor();
    tensor->SetDataType(DT_INT8);
    EXPECT_EQ(tensor->GetDataType(), DT_INT8);
}

TEST_F(TENSOR_UTest, Data)
{
    auto tensor = CpuKernelUtils::CreateTensor();
    int32_t input = 10;
    tensor->SetData(&input);
    tensor->SetDataSize(4);
    EXPECT_NE(tensor->GetData(), nullptr);
    EXPECT_EQ(tensor->GetDataSize(), 4);
}

TEST_F(TENSOR_UTest, GetTensorShapeFailed)
{
    auto shape = CpuKernelUtils::CreateTensorShape();
    shape->SetFormat(FORMAT_NCHW);
    vector<int64_t> dims = {3,1,3,5};
    shape->SetDimSizes(dims);
    shape->SetUnknownRank(false);
    auto tensor = CpuKernelUtils::CreateTensor();
    tensor->SetTensorShape(shape.get());

    MOCKER_CPP(CpuKernelUtils::CreateTensorShape, std::shared_ptr<TensorShape>(TensorShapeImpl *))
    .stubs()
    .will(returnValue(std::shared_ptr<TensorShape>(nullptr)));

    auto outShape = tensor->GetTensorShape();
}
