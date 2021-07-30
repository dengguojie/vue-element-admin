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

class TEST_EmbeddingLookup_UTest : public testing::Test {};

TEST_F(TEST_EmbeddingLookup_UTest, EmbeddingLookup_Success) {
  // raw data
  int expect_out[8] = {10, 11, 0, 0, 0, 0, 10, 11};
  int input_params[8] = {8, 9, 10, 11, 12, 13, 14, 15};
  int input_indices[4] = {5, 2, 8, 5};
  int offset[1] = {4};
  int output[8] = {0};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("EmbeddingLookup");

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {4, 2};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_INT32);
  inputTensor0->SetData(input_params);
  inputTensor0->SetDataSize(8 * sizeof(int));

  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {2, 2};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_INT32);
  inputTensor1->SetData(input_indices);
  inputTensor1->SetDataSize(4 * sizeof(int));

  auto inputTensor2 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor2, nullptr);
  auto aicpuShape2 = inputTensor2->GetTensorShape();
  std::vector<int64_t> shapes2 = {4};
  aicpuShape2->SetDimSizes(shapes2);
  inputTensor2->SetDataType(DT_INT32);
  inputTensor2->SetData(offset);
  inputTensor2->SetDataSize(1 * sizeof(int));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_INT32);
  outputTensor1->SetData(output);
  outputTensor1->SetDataSize(8 * sizeof(int));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);
  for (int i = 0; i < 8; ++i)
    std::cout << output[i] << " ";
  std::cout << std::endl;
  EXPECT_EQ(0, std::memcmp(output, expect_out, 8 * sizeof(int)));
}
