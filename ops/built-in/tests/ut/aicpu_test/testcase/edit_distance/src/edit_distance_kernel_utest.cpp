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

class TEST_EditDistance_UTest : public testing::Test {};

TEST_F(TEST_EditDistance_UTest, Int64Nor1) {
  // raw data
  int64_t hi[9] = {0, 0, 0, 1, 0, 1, 1, 1, 1};
  int64_t hv[3] = {1, 2, 3};
  int64_t hs[3] = {2, 2, 2};
  int64_t ti[12] = {0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1};
  int64_t tv[4] = {1, 2, 3, 1};
  int64_t ts[3] = {2, 2, 2};
  float out[4] = {0};
  float expect_out[4] = {1, 1, 1, 0};
  // create node def
  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("EditDistance");
  // set attr
  auto normalize = CpuKernelUtils::CreateAttrValue();
  normalize->SetBool(true);
  nodeDef->AddAttrs("normalize", normalize.get());
  // set input
  auto hypothesis_indices = nodeDef->AddInputs();
  EXPECT_NE(hypothesis_indices, nullptr);
  auto aicpuShape1 = hypothesis_indices->GetTensorShape();
  std::vector<int64_t> shapes1 = {3, 3};
  aicpuShape1->SetDimSizes(shapes1);
  hypothesis_indices->SetDataType(DT_INT64);
  hypothesis_indices->SetData(hi);
  hypothesis_indices->SetDataSize(9 * sizeof(int64_t));

  auto hypothesis_values = nodeDef->AddInputs();
  EXPECT_NE(hypothesis_values, nullptr);
  auto aicpuShape2 = hypothesis_values->GetTensorShape();
  std::vector<int64_t> shapes2 = {3};
  aicpuShape2->SetDimSizes(shapes2);
  hypothesis_values->SetDataType(DT_INT64);
  hypothesis_values->SetData(hv);
  hypothesis_values->SetDataSize(2 * sizeof(int64_t));

  auto hypothesis_shape = nodeDef->AddInputs();
  EXPECT_NE(hypothesis_shape, nullptr);
  auto aicpuShape3 = hypothesis_shape->GetTensorShape();
  std::vector<int64_t> shapes3 = {3};
  aicpuShape3->SetDimSizes(shapes3);
  hypothesis_shape->SetDataType(DT_INT64);
  hypothesis_shape->SetData(hs);
  hypothesis_shape->SetDataSize(3 * sizeof(int64_t));

  auto truth_indices = nodeDef->AddInputs();
  EXPECT_NE(truth_indices, nullptr);
  auto aicpuShape4 = truth_indices->GetTensorShape();
  std::vector<int64_t> shapes4 = {4, 3};
  aicpuShape4->SetDimSizes(shapes4);
  truth_indices->SetDataType(DT_INT64);
  truth_indices->SetData(ti);
  truth_indices->SetDataSize(12 * sizeof(int64_t));

  auto truth_values = nodeDef->AddInputs();
  EXPECT_NE(truth_values, nullptr);
  auto aicpuShape5 = truth_values->GetTensorShape();
  std::vector<int64_t> shapes5 = {4};
  aicpuShape5->SetDimSizes(shapes5);
  truth_values->SetDataType(DT_INT64);
  truth_values->SetData(tv);
  truth_values->SetDataSize(4 * sizeof(int64_t));

  auto truth_shape = nodeDef->AddInputs();
  EXPECT_NE(truth_shape, nullptr);
  auto aicpuShape6 = truth_shape->GetTensorShape();
  std::vector<int64_t> shapes6 = {3};
  aicpuShape6->SetDimSizes(shapes6);
  truth_shape->SetDataType(DT_INT64);
  truth_shape->SetData(ts);
  truth_shape->SetDataSize(3 * sizeof(int64_t));
  // set output
  auto outTensor = nodeDef->AddOutputs();
  EXPECT_NE(outTensor, nullptr);
  auto aicpuShape7 = outTensor->GetTensorShape();
  std::vector<int64_t> shapes7 = {2, 2};
  aicpuShape7->SetDimSizes(shapes7);
  outTensor->SetDataType(DT_FLOAT);
  outTensor->SetData(out);
  outTensor->SetDataSize(4 * sizeof(float));

  CpuKernelContext ctx(DEVICE);
  using namespace std;
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);
  EXPECT_EQ(0, std::memcmp(out, expect_out, 4 * sizeof(float)));
}
