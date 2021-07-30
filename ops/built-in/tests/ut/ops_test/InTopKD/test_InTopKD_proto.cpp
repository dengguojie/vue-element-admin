#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class InTopKDTest : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "in_top_k_d test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "in_top_k_d test TearDown" << std::endl;
  }
};

TEST_F(InTopKDTest, in_top_k_d_test_case_1) {
  ge::op::InTopKD op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({16, 24});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({16, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 10;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(InTopKDTest, in_top_k_d_test_case_2) {
  ge::op::InTopKD op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({52, 1050});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({52, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 400;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {52};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(InTopKDTest, in_top_k_d_test_case_3) {
  ge::op::InTopKD op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({-1, 234});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({-1, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 100;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(InTopKDTest, in_top_k_d_test_case_4) {
  ge::op::InTopKD op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({38, 23344});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({38, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 10000;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {38};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(InTopKDTest, in_top_k_d_test_case_5) {
  ge::op::InTopKD op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({38, 24});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({38, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 100;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {38};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(InTopKDTest, in_top_k_d_test_case_6) {
  ge::op::InTopKD op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({18, 78});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({28, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 10;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(InTopKDTest, in_top_k_d_test_case_7) {
  ge::op::InTopKD op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({18, 78, 98});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({18, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 10;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

