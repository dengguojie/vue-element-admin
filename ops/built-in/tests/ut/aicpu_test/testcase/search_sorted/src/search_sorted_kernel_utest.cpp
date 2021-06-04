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

class TEST_SearchSorted_UTest : public testing::Test {};

#define ADD_CASE(aicpu_type, base_type, out_type, out_dtype)             \
  TEST_F(TEST_SearchSorted_UTest,                                        \
         SearchSorted_Success_##aicpu_type##_##out_type) {               \
    aicpu_type sorted_sequence[2 * 5] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}; \
    aicpu_type values[2 * 3] = {3, 6, 9, 3, 6, 9};                       \
    out_type output[2 * 3] = {0};                                        \
    out_type expected_output[2 * 3] = {1, 3, 4, 1, 2, 4};                \
    auto nodeDef = CpuKernelUtils::CreateNodeDef();                      \
    nodeDef->SetOpType("SearchSorted");                                  \
    auto right = CpuKernelUtils::CreateAttrValue();                      \
    right->SetBool(false);                                               \
    nodeDef->AddAttrs("right", right.get());                             \
    auto inputTensor0 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor0, nullptr);                                    \
    auto aicpuShape0 = inputTensor0->GetTensorShape();                   \
    std::vector<int64_t> shapes0 = {2, 5};                               \
    aicpuShape0->SetDimSizes(shapes0);                                   \
    inputTensor0->SetDataType(base_type);                                \
    inputTensor0->SetData(sorted_sequence);                              \
    inputTensor0->SetDataSize(2 * 5 * sizeof(aicpu_type));               \
    auto inputTensor1 = nodeDef->AddInputs();                            \
    EXPECT_NE(inputTensor1, nullptr);                                    \
    auto aicpuShape1 = inputTensor1->GetTensorShape();                   \
    std::vector<int64_t> shapes1 = {2, 3};                               \
    aicpuShape1->SetDimSizes(shapes1);                                   \
    inputTensor1->SetDataType(base_type);                                \
    inputTensor1->SetData(values);                                       \
    inputTensor1->SetDataSize(2 * 3 * sizeof(aicpu_type));               \
    auto outputTensor1 = nodeDef->AddOutputs();                          \
    EXPECT_NE(outputTensor1, nullptr);                                   \
    auto aicpuShape3 = outputTensor1->GetTensorShape();                  \
    std::vector<int64_t> shapes2 = {2, 3};                               \
    aicpuShape3->SetDimSizes(shapes2);                                   \
    outputTensor1->SetDataType(out_dtype);                               \
    outputTensor1->SetData(output);                                      \
    outputTensor1->SetDataSize(2 * 3 * sizeof(out_type));                \
    CpuKernelContext ctx(DEVICE);                                        \
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);                \
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);      \
    EXPECT_EQ(ret, KERNEL_STATUS_OK);                                    \
    EXPECT_EQ(0, std::memcmp(output, expected_output, sizeof(output)));  \
  }

TEST_F(TEST_SearchSorted_UTest,
       SearchSorted_Input_Type_Error) {
  uint32_t sorted_sequence[2 * 5] = { 1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  uint32_t values[2 * 3] = {3, 6, 9, 3, 6, 9};
  int32_t output[2 * 3] = {0};
  int32_t expected_output[2 * 3] = {1, 3, 4, 1, 2, 4};
  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("SearchSorted");
  auto right = CpuKernelUtils::CreateAttrValue();
  right->SetBool(false);
  nodeDef->AddAttrs("right", right.get());
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {2, 5};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_UINT32);
  inputTensor0->SetData(sorted_sequence);
  inputTensor0->SetDataSize(2 * 5 * sizeof(uint32_t));
  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {2, 3};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_UINT32);
  inputTensor1->SetData(values);
  inputTensor1->SetDataSize(2 * 3 * sizeof(uint32_t));
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  auto aicpuShape3 = outputTensor1->GetTensorShape();
  std::vector<int64_t> shapes2 = {2, 3};
  aicpuShape3->SetDimSizes(shapes2);
  outputTensor1->SetDataType(DT_INT32);
  outputTensor1->SetData(output);
  outputTensor1->SetDataSize(2 * 3 * sizeof(int32_t));
  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest,
       SearchSorted_Input_Unsorted_Error) {
  float sorted_sequence[2 * 5] = { 9, 3, 5, 7, 9, 2, 4, 6, 8, 1};
  float values[2 * 3] = {3, 6, 9, 3, 6, 9};
  int32_t output[2 * 3] = {0};
  int32_t expected_output[2 * 3] = {1, 3, 4, 1, 2, 4};
  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("SearchSorted");
  auto right = CpuKernelUtils::CreateAttrValue();
  right->SetBool(false);
  nodeDef->AddAttrs("right", right.get());
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {2, 5};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_FLOAT);
  inputTensor0->SetData(sorted_sequence);
  inputTensor0->SetDataSize(2 * 5 * sizeof(float));
  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {2, 3};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_FLOAT);
  inputTensor1->SetData(values);
  inputTensor1->SetDataSize(2 * 3 * sizeof(float));
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  auto aicpuShape3 = outputTensor1->GetTensorShape();
  std::vector<int64_t> shapes2 = {2, 3};
  aicpuShape3->SetDimSizes(shapes2);
  outputTensor1->SetDataType(DT_INT32);
  outputTensor1->SetData(output);
  outputTensor1->SetDataSize(2 * 3 * sizeof(int32_t));
  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

ADD_CASE(float, DT_FLOAT, int, DT_INT32)

ADD_CASE(double, DT_DOUBLE, int, DT_INT32)

ADD_CASE(int32_t, DT_INT32, int, DT_INT32)

ADD_CASE(int64_t, DT_INT64, int, DT_INT32)

ADD_CASE(float, DT_FLOAT, int64_t, DT_INT64)

ADD_CASE(double, DT_DOUBLE, int64_t, DT_INT64)

ADD_CASE(int32_t, DT_INT32, int64_t, DT_INT64)

ADD_CASE(int64_t, DT_INT64, int64_t, DT_INT64)