#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_DynamicStitch_UTest : public testing::Test {};

#define ADD_CASE(aicpu_type, base_type)                                  \
  TEST_F(TEST_DynamicStitch_UTest, DynamicStitch_Success_##aicpu_type) { \
    int indices0[1] = {6};                                               \
    int indices1[2] = {4, 1};                                            \
    int indices2[2 * 2] = {5, 2, 0, 3};                                  \
    aicpu_type data0[1 * 2] = {61, 62};                                  \
    aicpu_type data1[2 * 2] = {41, 42, 11, 12};                          \
    aicpu_type data2[2 * 2 * 2] = {51, 52, 21, 22, 1, 2, 31, 32};        \
    aicpu_type merged[7 * 2] = {0};                                      \
    aicpu_type expected_merged[7 * 2] = {1,  2,  11, 12, 21, 22, 31,     \
                                         32, 41, 42, 51, 52, 61, 62};    \
    auto nodeDef = CpuKernelUtils::CreateNodeDef();                      \
    nodeDef->SetOpType("DynamicStitch");                                 \
    auto inputTensor0 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor0, nullptr);                                    \
    auto aicpuShape0 = inputTensor0->GetTensorShape();                   \
    std::vector<int64_t> shapes0 = {1};                                  \
    aicpuShape0->SetDimSizes(shapes0);                                   \
    inputTensor0->SetDataType(DT_INT32);                                 \
    inputTensor0->SetData(indices0);                                     \
    inputTensor0->SetDataSize(1 * sizeof(int));                          \
    auto inputTensor1 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor1, nullptr);                                    \
    auto aicpuShape1 = inputTensor1->GetTensorShape();                   \
    std::vector<int64_t> shapes1 = {2};                                  \
    aicpuShape1->SetDimSizes(shapes1);                                   \
    inputTensor1->SetDataType(DT_INT32);                                 \
    inputTensor1->SetData(indices1);                                     \
    inputTensor1->SetDataSize(2 * sizeof(int));                          \
    auto inputTensor2 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor2, nullptr);                                    \
    auto aicpuShape2 = inputTensor2->GetTensorShape();                   \
    std::vector<int64_t> shapes2 = {2, 2};                               \
    aicpuShape2->SetDimSizes(shapes2);                                   \
    inputTensor2->SetDataType(DT_INT32);                                 \
    inputTensor2->SetData(indices2);                                     \
    inputTensor2->SetDataSize(2 * 2 * sizeof(int));                      \
    auto inputTensor3 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor3, nullptr);                                    \
    auto aicpuShape3 = inputTensor3->GetTensorShape();                   \
    std::vector<int64_t> shapes3 = {1, 2};                               \
    aicpuShape3->SetDimSizes(shapes3);                                   \
    inputTensor3->SetDataType(base_type);                                \
    inputTensor3->SetData(data0);                                        \
    inputTensor3->SetDataSize(1 * 2 * sizeof(aicpu_type));               \
    auto inputTensor4 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor4, nullptr);                                    \
    auto aicpuShape4 = inputTensor4->GetTensorShape();                   \
    std::vector<int64_t> shapes4 = {2, 2};                               \
    aicpuShape4->SetDimSizes(shapes4);                                   \
    inputTensor4->SetDataType(base_type);                                \
    inputTensor4->SetData(data1);                                        \
    inputTensor4->SetDataSize(2 * 2 * sizeof(aicpu_type));               \
    auto inputTensor5 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor5, nullptr);                                    \
    auto aicpuShape5 = inputTensor5->GetTensorShape();                   \
    std::vector<int64_t> shapes5 = {2, 2, 2};                            \
    aicpuShape5->SetDimSizes(shapes5);                                   \
    inputTensor5->SetDataType(base_type);                                \
    inputTensor5->SetData(data2);                                        \
    inputTensor5->SetDataSize(2 * 2 * 2 * sizeof(aicpu_type));           \
    auto outputTensor1 = nodeDef->AddOutputs();                          \
    EXPECT_NE(outputTensor1, nullptr);                                   \
    outputTensor1->SetDataType(base_type);                               \
    outputTensor1->SetData(merged);                                      \
    outputTensor1->SetDataSize(7 * 2 * sizeof(aicpu_type));              \
    CpuKernelContext ctx(DEVICE);                                        \
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);                \
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);      \
    EXPECT_EQ(ret, KERNEL_STATUS_OK);                                    \
    EXPECT_EQ(0, std::memcmp(merged, expected_merged, sizeof(merged)));  \
  }

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(int, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint32_t, DT_UINT32)

ADD_CASE(uint64_t, DT_UINT64)