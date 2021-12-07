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

class TEST_NMSWithMask_UTest : public testing::Test {};

TEST_F(TEST_NMSWithMask_UTest, NMSWithMask_Success) {
  // raw data
  float box_scores[4*5] = {100.0, 100.0, 150.0, 168.0, 0.63, \
                           150.0, 75.0, 315.0, 190.0, 0.55,  \
                           12.0, 190.0, 300.0, 390.0, 0.9,   \
                           28.0, 130.0, 134.0, 302.0, 0.3};
  float select_boxes[4*5] = {0};
  int select_ids[4] = {0};
  bool select_mask[4] = {0};

  float select_boxes_expect[4*5] = {12.0, 190.0, 300.0, 390.0, 0.9,   \
                                    100.0, 100.0, 150.0, 168.0, 0.63, \
                                    150.0, 75.0, 315.0, 190.0, 0.55,  \
                                    0.0, 0.0, 0.0, 0.0, 0.0};
  int select_ids_expect[4] = {2, 0, 1, 0};

  bool select_mask_expect[4] = {true, true, true, false};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("NMSWithMask");

  // set attr
  auto iou_threshold = CpuKernelUtils::CreateAttrValue();
  iou_threshold->SetFloat(0.1);
  nodeDef->AddAttrs("iou_threshold", iou_threshold.get());

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {4, 5};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_FLOAT);
  inputTensor0->SetData(box_scores);
  inputTensor0->SetDataSize(4 * 5 * sizeof(float));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_FLOAT);
  outputTensor1->SetData(select_boxes);
  outputTensor1->SetDataSize(4 * 5 * sizeof(float));

  auto outputTensor2 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor2, nullptr);
  outputTensor2->SetDataType(DT_INT32);
  outputTensor2->SetData(select_ids);
  outputTensor2->SetDataSize(4 * sizeof(int));

  auto outputTensor3 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor3, nullptr);
  outputTensor3->SetDataType(DT_BOOL);
  outputTensor3->SetData(select_mask);
  outputTensor3->SetDataSize(4 * sizeof(bool));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);

  for (int i = 0; i < 4*5; ++i) {
    EXPECT_EQ(select_boxes[i], select_boxes_expect[i]);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(select_ids[i], select_ids_expect[i]);
    EXPECT_EQ(select_mask[i], select_mask_expect[i]);
  }
}
