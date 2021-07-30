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

class TEST_ComputeAccidentalHits_UTest : public testing::Test {};

TEST_F(TEST_ComputeAccidentalHits_UTest, ComputeAccidentalHits_Success) {
  // raw data
  int x[3][2] = {{1, 2}, {0, 4}, {3, 3}};
  int sampled_candidates[5] = {0, 1, 2, 3, 4};
  int acc_indices[6] = {0};
  int acc_ids[6] = {0};
  float acc_weights[6] = {0};

  int acc_indices_expect[6] = {0, 0, 1, 1, 2, 2};
  int acc_ids_expect[6] = {1, 2, 0, 4, 3, 3};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("ComputeAccidentalHits");

  // set attr
  auto num_true = CpuKernelUtils::CreateAttrValue();
  num_true->SetInt(2);
  nodeDef->AddAttrs("num_true", num_true.get());

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {3, 2};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_INT32);
  inputTensor0->SetData(x);
  inputTensor0->SetDataSize(3 * 2 * sizeof(int));

  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {5};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_INT32);
  inputTensor1->SetData(sampled_candidates);
  inputTensor1->SetDataSize(5 * sizeof(int));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_INT32);
  outputTensor1->SetData(acc_indices);
  outputTensor1->SetDataSize(6 * sizeof(int));

  auto outputTensor2 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor2, nullptr);
  outputTensor2->SetDataType(DT_INT32);
  outputTensor2->SetData(acc_ids);
  outputTensor2->SetDataSize(6 * sizeof(int));

  auto outputTensor3 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor3, nullptr);
  outputTensor3->SetDataType(DT_FLOAT);
  outputTensor3->SetData(acc_weights);
  outputTensor3->SetDataSize(6 * sizeof(float));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);

  std::cout << "************output*************" << std::endl;
  for (int i = 0; i < 6; ++i) {
    std::cout << acc_indices[i] << ", " << acc_ids[i] << ", " << acc_weights[i] << std::endl;
    EXPECT_EQ(acc_indices[i], acc_indices_expect[i]);
    EXPECT_EQ(acc_weights[i], static_cast<float>(-__FLT_MAX__));
  }
}
