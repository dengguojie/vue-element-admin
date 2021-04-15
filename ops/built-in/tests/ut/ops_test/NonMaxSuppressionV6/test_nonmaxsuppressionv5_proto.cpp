#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "image_ops.h"

class NonMaxSuppressionV5Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonMaxSuppressionV5 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonMaxSuppressionV5 TearDown" << std::endl;
  }
};

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_1) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 1, 6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_2) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2, 1, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 1, 6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_3) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_4) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_5) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({3}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_6) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({3}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_7) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
   op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_8) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_9) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.SetAttr("pad_to_max_output_size",true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_10) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2,4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.SetAttr("pad_to_max_output_size",true);
  op.SetAttr("T", ge::DT_FLOAT);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_11) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2,4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.SetAttr("pad_to_max_output_size",true);
  op.SetAttr("T", ge::DT_FLOAT);
  
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {-1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto concat_dim = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_max_output_size(concat_dim);
  auto desc = op.GetInputDesc("max_output_size");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("max_output_size", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV5Test, non_max_suppressio_test_case_12) {
  ge::op::NonMaxSuppressionV5 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2,4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("score_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("soft_nms_sigma", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.SetAttr("pad_to_max_output_size", false);
  op.SetAttr("T", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}