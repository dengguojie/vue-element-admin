#include <gtest/gtest.h>

#include <vector>

#include "nn_norm_ops.h"
#include "op_proto_test_util.h"

class MultilabelMarginLossTest : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "multilabel_margin_loss test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "multilabel_margin_loss test TearDown" << std::endl; }
};

TEST_F(MultilabelMarginLossTest, multilabel_margin_loss_test_case_1) {
  // define op
  ge::op::MultilabelMarginLoss multilabel_margin_loss_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({2, 4});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  // update op input
  multilabel_margin_loss_op.UpdateInputDesc("x", tensorDesc);
  multilabel_margin_loss_op.UpdateInputDesc("target", tensorDesc);

  // call InferShapeAndType
  auto ret = multilabel_margin_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // compare dtype and shape of op output
  auto output_desc = multilabel_margin_loss_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MultilabelMarginLossTest, multilabel_margin_loss_test_case_2) {
  // define op
  ge::op::MultilabelMarginLoss multilabel_margin_loss_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({1, 4});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  // update op input
  multilabel_margin_loss_op.UpdateInputDesc("x", tensorDesc);
  multilabel_margin_loss_op.UpdateInputDesc("target", tensorDesc);

  // call InferShapeAndType
  auto ret = multilabel_margin_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // compare dtype and shape of op output
  auto output_desc = multilabel_margin_loss_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MultilabelMarginLossTest, multilabel_margin_loss_test_case_3) {
  // define op
  ge::op::MultilabelMarginLoss multilabel_margin_loss_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({1, 1});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  // update op input
  multilabel_margin_loss_op.UpdateInputDesc("x", tensorDesc);
  multilabel_margin_loss_op.UpdateInputDesc("target", tensorDesc);

  // call InferShapeAndType
  auto ret = multilabel_margin_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // compare dtype and shape of op output
  auto output_desc = multilabel_margin_loss_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MultilabelMarginLossTest, multilabel_margin_loss_test_case_4) {
  // define op
  ge::op::MultilabelMarginLoss multilabel_margin_loss_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({1});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  // update op input
  multilabel_margin_loss_op.UpdateInputDesc("x", tensorDesc);
  multilabel_margin_loss_op.UpdateInputDesc("target", tensorDesc);

  // call InferShapeAndType
  auto ret = multilabel_margin_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // compare dtype and shape of op output
  auto output_desc = multilabel_margin_loss_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MultilabelMarginLossTest, multilabel_margin_loss_test_case_5) {
  // define op
  ge::op::MultilabelMarginLoss multilabel_margin_loss_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({1000});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  // update op input
  multilabel_margin_loss_op.UpdateInputDesc("x", tensorDesc);
  multilabel_margin_loss_op.UpdateInputDesc("target", tensorDesc);

  // call InferShapeAndType
  auto ret = multilabel_margin_loss_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // compare dtype and shape of op output
  auto output_desc = multilabel_margin_loss_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}