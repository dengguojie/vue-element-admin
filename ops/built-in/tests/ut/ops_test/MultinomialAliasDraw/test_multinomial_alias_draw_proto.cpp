#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "random_ops.h"

class multinomial_alias_draw : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "multinomial_alias_draw SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "multinomial_alias_draw TearDown" << std::endl;
  }
};

TEST_F(multinomial_alias_draw, multinomial_alias_draw_infer_shape_fp32) {
  ge::op::MultinomialAliasDraw op;
  ge::TensorDesc tensor_j_desc;
  ge::TensorDesc tensor_q_desc;
  ge::Shape shape({100});
  tensor_j_desc.SetDataType(ge::DT_INT64);
  tensor_j_desc.SetShape(shape);
  tensor_j_desc.SetOriginShape(shape);
  tensor_q_desc.SetDataType(ge::DT_FLOAT);
  tensor_q_desc.SetShape(shape);
  tensor_q_desc.SetOriginShape(shape);
  op.UpdateInputDesc("j", tensor_j_desc);
  op.UpdateInputDesc("q", tensor_q_desc);
  op.SetAttr("num_samples", 50);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {50};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(multinomial_alias_draw, multinomial_alias_draw_infer_shape_fp64) {
  ge::op::MultinomialAliasDraw op;
  ge::TensorDesc tensor_j_desc;
  ge::TensorDesc tensor_q_desc;
  ge::Shape shape({100});
  tensor_j_desc.SetDataType(ge::DT_INT64);
  tensor_j_desc.SetShape(shape);
  tensor_j_desc.SetOriginShape(shape);
  tensor_q_desc.SetDataType(ge::DT_DOUBLE);
  tensor_q_desc.SetShape(shape);
  tensor_q_desc.SetOriginShape(shape);
  op.UpdateInputDesc("j", tensor_j_desc);
  op.UpdateInputDesc("q", tensor_q_desc);
  op.SetAttr("num_samples", 100);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {100};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
