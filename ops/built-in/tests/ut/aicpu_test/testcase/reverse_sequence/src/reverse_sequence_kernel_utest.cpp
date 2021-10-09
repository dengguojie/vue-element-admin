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

class TEST_ReverseSequence_UTest : public testing::Test {};

TEST_F(TEST_ReverseSequence_UTest, ReverseSequence_Success) {
  // raw data
  float x[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  uint64_t seq_lengths[3] = {1, 2, 3};
  float y[3][3] = {0};
  float y_expect[3][3] = {{1, 5, 9}, {4, 2, 6}, {7, 8, 3}};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("ReverseSequence");

  // set attr
  auto seq_dim = CpuKernelUtils::CreateAttrValue();
  seq_dim->SetInt(0);
  nodeDef->AddAttrs("seq_dim", seq_dim.get());

  auto batch_dim = CpuKernelUtils::CreateAttrValue();
  batch_dim->SetInt(1);
  nodeDef->AddAttrs("batch_dim", batch_dim.get());

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {3, 3};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_FLOAT);
  inputTensor0->SetData(x);
  inputTensor0->SetDataSize(3 * 3 * sizeof(float));

  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {3};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_INT64);
  inputTensor1->SetData(seq_lengths);
  inputTensor1->SetDataSize(3 * sizeof(uint64_t));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_FLOAT);
  outputTensor1->SetData(y);
  outputTensor1->SetDataSize(3 * sizeof(int));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);

  float eps = 0.0001;
  std::cout << "************y*************" << std::endl;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << y[i][j] << " ";
      EXPECT_LT(std::abs(y[i][j] - y_expect[i][j]), eps);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
