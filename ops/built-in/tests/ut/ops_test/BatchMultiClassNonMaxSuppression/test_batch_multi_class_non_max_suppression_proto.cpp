#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class BatchMultiClassNonMaxSuppression : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BatchMultiClassNonMaxSuppression SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BatchMultiClassNonMaxSuppression TearDown" << std::endl;
  }
};

TEST_F(BatchMultiClassNonMaxSuppression, _infershape_fasterrcnn) {
  int64_t batchSize = 1;
  int64_t outputNum = 100; 
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({batchSize, 1024, 1, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {batchSize, 1024, 1, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({batchSize, 1024, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {batchSize, 1024, 1}, ge::FORMAT_ND));
  op.SetAttr("score_threshold", (float)0.3);
  op.SetAttr("iou_threshold", (float)0.6);
  op.SetAttr("max_size_per_class", 100);
  op.SetAttr("max_total_size", outputNum);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("nmsed_boxes");
  std::vector<int64_t> expected_var_output_shape = {batchSize, outputNum, 4};
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

  out_var_desc = op.GetOutputDesc("nmsed_scores");
  expected_var_output_shape = {batchSize, outputNum};
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

  out_var_desc = op.GetOutputDesc("nmsed_classes");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

  out_var_desc = op.GetOutputDesc("nmsed_num");
  expected_var_output_shape = {batchSize};
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(BatchMultiClassNonMaxSuppression, InfershapeBatchMultiClassNonMax_001) {
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.UpdateInputDesc(
      "boxes", create_desc_with_ori({1, 1024, 1, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 1, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores",
                     create_desc_with_ori({1, 1024, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 1}, ge::FORMAT_ND));
  op.SetAttr("transpose_box", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BatchMultiClassNonMaxSuppression, InfershapeBatchMultiClassNonMax_002) {
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores",
                     create_desc_with_ori({1, 1024, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 1}, ge::FORMAT_ND));
  op.SetAttr("transpose_box", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BatchMultiClassNonMaxSuppression, InfershapeBatchMultiClassNonMax_003) {
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.UpdateInputDesc(
      "boxes", create_desc_with_ori({1, 1024, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 1, 1}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores",
                     create_desc_with_ori({1, 1024, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 1}, ge::FORMAT_ND));
  op.SetAttr("transpose_box", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BatchMultiClassNonMaxSuppression, InfershapeBatchMultiClassNonMax_004) {
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.UpdateInputDesc(
      "boxes", create_desc_with_ori({1, 1024, 1, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 1, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.SetAttr("transpose_box", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BatchMultiClassNonMaxSuppression, InfershapeBatchMultiClassNonMax_005) {
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.UpdateInputDesc(
      "boxes", create_desc_with_ori({1, 1024, 2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores",
                     create_desc_with_ori({1, 1024, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1024, 1}, ge::FORMAT_ND));

  op.SetAttr("transpose_box", false);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BatchMultiClassNonMaxSuppression, InfershapeBatchMultiClassNonMax_006) {
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.SetAttr("transpose_box", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}