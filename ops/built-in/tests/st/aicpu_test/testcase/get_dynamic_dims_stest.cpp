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

class GET_DYNAMIC_DIMS_KERNEL_ST : public testing::Test {
protected:
    virtual void SetUp() {}

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

TEST_F(GET_DYNAMIC_DIMS_KERNEL_ST, Inputs_INT32)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("GetDynamicDims");

    // set input1
    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);
    std::vector<int32_t> x1{ 3, 2, 4, 1 };
    auto x1Shape = x1Tensor->GetTensorShape();
    x1Shape->SetDimSizes({4});
    x1Tensor->SetDataType(DT_INT32);
    x1Tensor->SetData(x1.data());

    // set input2
    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);
    std::vector<int32_t> x2{ 1, 2, 1 };
    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes({3});
    x2Tensor->SetDataType(DT_INT32);
    x2Tensor->SetData(x2.data());

    // set input3
    auto x3Tensor = nodeDef->AddInputs();
    EXPECT_NE(x3Tensor, nullptr);
    std::vector<int32_t> x3{ 16, 112, 112, 3, 4 };
    auto x3Shape = x3Tensor->GetTensorShape();
    x3Shape->SetDimSizes({5});
    x3Tensor->SetDataType(DT_INT32);
    x3Tensor->SetData(x3.data());

    // set attrs
    int64_t N = 3;
    auto NAttr = CpuKernelUtils::CreateAttrValue();
    NAttr->SetInt(N);
    nodeDef->AddAttrs("N", NAttr.get());

    vector<int64_t> shape_info{ 4, 3, 2, -1, 1, 3, 1, 2, 1, 5, 16, -1, -1, 3, 4 };
    auto shapeAttrs = CpuKernelUtils::CreateAttrValue();
    shapeAttrs->SetListInt(shape_info);
    nodeDef->AddAttrs("shape_info", shapeAttrs.get());

    // set output
    auto dimsTensor = nodeDef->AddOutputs();
    EXPECT_NE(dimsTensor, nullptr);
    std::vector<int64_t> dims(3);
    auto dimsShape = dimsTensor->GetTensorShape();
    dimsShape->SetDimSizes({3});
    dimsTensor->SetDataType(DT_INT64);
    dimsTensor->SetData(dims.data());
    dimsTensor->SetDataSize(3 * sizeof(int64_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    std::cout << "dims: [" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]" << std::endl;
}

TEST_F(GET_DYNAMIC_DIMS_KERNEL_ST, Inputs_INT64)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("GetDynamicDims");

    // set input1
    auto x1Tensor = nodeDef->AddInputs();
    EXPECT_NE(x1Tensor, nullptr);
    std::vector<int64_t> x1{ 3, 2, 4, 1 };
    auto x1Shape = x1Tensor->GetTensorShape();
    x1Shape->SetDimSizes({4});
    x1Tensor->SetDataType(DT_INT64);
    x1Tensor->SetData(x1.data());

    // set input2
    auto x2Tensor = nodeDef->AddInputs();
    EXPECT_NE(x2Tensor, nullptr);
    std::vector<int64_t> x2{ 1, 2, 1 };
    auto x2Shape = x2Tensor->GetTensorShape();
    x2Shape->SetDimSizes({3});
    x2Tensor->SetDataType(DT_INT64);
    x2Tensor->SetData(x2.data());

    // set input3
    auto x3Tensor = nodeDef->AddInputs();
    EXPECT_NE(x3Tensor, nullptr);
    std::vector<int64_t> x3{ 16, 112, 112, 3, 4 };
    auto x3Shape = x3Tensor->GetTensorShape();
    x3Shape->SetDimSizes({5});
    x3Tensor->SetDataType(DT_INT64);
    x3Tensor->SetData(x3.data());

    // set attrs
    int64_t N = 3;
    auto NAttr = CpuKernelUtils::CreateAttrValue();
    NAttr->SetInt(N);
    nodeDef->AddAttrs("N", NAttr.get());

    vector<int64_t> shape_info{ 4, 3, 2, -1, 1, 3, 1, 2, 1, 5, 16, -1, -1, 3, 4 };
    auto shapeAttrs = CpuKernelUtils::CreateAttrValue();
    shapeAttrs->SetListInt(shape_info);
    nodeDef->AddAttrs("shape_info", shapeAttrs.get());

    // set output
    auto dimsTensor = nodeDef->AddOutputs();
    EXPECT_NE(dimsTensor, nullptr);
    std::vector<int64_t> dims(3);
    auto dimsShape = dimsTensor->GetTensorShape();
    dimsShape->SetDimSizes({3});
    dimsTensor->SetDataType(DT_INT64);
    dimsTensor->SetData(dims.data());
    dimsTensor->SetDataSize(3 * sizeof(int64_t));

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    std::cout << "dims: [" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]" << std::endl;
}
