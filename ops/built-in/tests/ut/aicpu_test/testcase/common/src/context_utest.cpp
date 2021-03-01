#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif

#include "cpu_node_def.h"
#include "cpu_tensor.h"
#include "cpu_context.h"
#include "cpu_kernel_utils.h"
#include "device.h"

using namespace std;
using namespace aicpu;

class CONTEXT_UTest : public testing::Test {};

TEST_F(CONTEXT_UTest, InitHost)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("test");

    auto inputTensor = nodeDef->AddInputs();
    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {3,18};
    aicpuShape->SetDimSizes(shapes);
    inputTensor->SetDataType(DT_INT8);
    auto outputTensor = nodeDef->AddOutputs();

    auto attr = CpuKernelUtils::CreateAttrValue();
    for (auto shape : shapes) {
        attr->AddListInt(shape);
    }
    nodeDef->AddAttrs("shape", attr.get());

    CpuKernelContext ctx(HOST);
    ctx.Init(nodeDef.get());
    string op = ctx.GetOpType();
    EXPECT_EQ(op, "test");

    Tensor *input0 = ctx.Input(0);
    EXPECT_NE(input0, nullptr);
    Tensor *input1 = ctx.Input(1);
    EXPECT_EQ(input1, nullptr);

    Tensor *output0 = ctx.Output(0);
    EXPECT_NE(output0, nullptr);
    Tensor *output1 = ctx.Output(1);
    EXPECT_EQ(output1, nullptr);

    AttrValue *attrShape = ctx.GetAttr("shape");
    EXPECT_NE(attrShape, nullptr);  

    uint32_t inputsSize = ctx.GetInputsSize();
    EXPECT_EQ(inputsSize, 1);
    uint32_t outputsSize = ctx.GetOutputsSize();
    EXPECT_EQ(outputsSize, 1);
}

TEST_F(CONTEXT_UTest, InitDevice)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("test");

    auto inputTensor = nodeDef->AddInputs();
    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {3,18};
    aicpuShape->SetDimSizes(shapes);
    inputTensor->SetDataType(DT_INT8);
    auto outputTensor = nodeDef->AddOutputs();

    auto attr = CpuKernelUtils::CreateAttrValue();
    for (auto shape : shapes) {
        attr->AddListInt(shape);
    }
    nodeDef->AddAttrs("shape", attr.get());

    CpuKernelContext ctx(DEVICE);
    ctx.Init(nodeDef.get());
    string op = ctx.GetOpType();
    EXPECT_EQ(op, "test");

    Tensor *input0 = ctx.Input(0);
    EXPECT_NE(input0, nullptr);
    Tensor *input1 = ctx.Input(1);
    EXPECT_EQ(input1, nullptr);

    Tensor *output0 = ctx.Output(0);
    EXPECT_NE(output0, nullptr);
    Tensor *output1 = ctx.Output(1);
    EXPECT_EQ(output1, nullptr);

    AttrValue *attrShape = ctx.GetAttr("shape");
    EXPECT_NE(attrShape, nullptr);

    attrShape = ctx.GetAttr("aaa");
    EXPECT_EQ(attrShape, nullptr);  

    uint32_t inputsSize = ctx.GetInputsSize();
    EXPECT_EQ(inputsSize, 1);
    uint32_t outputsSize = ctx.GetOutputsSize();
    EXPECT_EQ(outputsSize, 1);
}

TEST_F(CONTEXT_UTest, ParallelFor)
{
    CpuKernelContext ctx(DEVICE);
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, 10, 1, [&](int64_t start, int64_t end){});
    EXPECT_EQ(ret, 0);
}

TEST_F(CONTEXT_UTest, GetCpuNumDeviceSuccess)
{
    CpuKernelContext ctx(DEVICE);
    uint32_t ret = CpuKernelUtils::GetCPUNum(ctx);
    EXPECT_NE(ret, 0);
}

TEST_F(CONTEXT_UTest, GetCpuNumDeviceFailed1)
{
    CpuKernelContext ctx(DEVICE);
    ctx.device_ = nullptr;
    uint32_t ret = CpuKernelUtils::GetCPUNum(ctx);
    EXPECT_EQ(ret, 0);
}

TEST_F(CONTEXT_UTest, GetCpuNumDeviceFailed2)
{
    CpuKernelContext ctx(DEVICE);
    Sharder *sharder = ctx.device_->sharder_;
    ctx.device_->sharder_ = nullptr;
    uint32_t ret = CpuKernelUtils::GetCPUNum(ctx);
    ctx.device_->sharder_ = sharder;
    EXPECT_EQ(ret, 0);
}

TEST_F(CONTEXT_UTest, GetCpuNumHost)
{
    CpuKernelContext ctx(HOST);
    uint32_t ret = CpuKernelUtils::GetCPUNum(ctx);
    EXPECT_NE(ret, 0);
}
