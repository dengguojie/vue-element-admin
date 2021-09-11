#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class InTopKTest : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "in_top_k test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "in_top_k test TearDown" << std::endl;
  }
};

TEST_F(InTopKTest, in_top_k_test_case_1) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({16, 24});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({16, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

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

TEST_F(InTopKTest, in_top_k_test_case_2) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({52, 1050});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({52, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

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

TEST_F(InTopKTest, in_top_k_test_case_3) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({-1, 234});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({-1, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

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

TEST_F(InTopKTest, in_top_k_test_case_4) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({38, 23344});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({38, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

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

TEST_F(InTopKTest, in_top_k_test_case_5) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({38, 24});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({38, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

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

TEST_F(InTopKTest, in_top_k_test_case_6) {
  ge::op::InTopK op;

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

TEST_F(InTopKTest, in_top_k_test_case_7) {
  ge::op::InTopK op;

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

TEST_F(InTopKTest, in_top_k_test_case_8) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({56, 789});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({56, 63});
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 10;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(InTopKTest, in_top_k_test_case_9) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({-2, });
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({-1, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 10;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(InTopKTest, in_top_k_test_case_10) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({958, 9857});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({958, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 3000;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {958};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(InTopKTest, in_top_k_test_case_11) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({1312, 4});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({1312, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 2;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {1312};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(InTopKTest, in_top_k_test_case_12) {
  ge::op::InTopK op;

  ge::TensorDesc tensorDesc1;
  ge::Shape shape1({795, 35208});
  tensorDesc1.SetDataType(ge::DT_FLOAT);
  tensorDesc1.SetShape(shape1);
  tensorDesc1.SetOriginShape(shape1);

  ge::TensorDesc tensorDesc2;
  ge::Shape shape2({795, });
  tensorDesc2.SetDataType(ge::DT_INT32);
  tensorDesc2.SetShape(shape2);
  tensorDesc2.SetOriginShape(shape2);

  op.UpdateInputDesc("x1", tensorDesc1);
  op.UpdateInputDesc("x2", tensorDesc2);
  auto k = 20039;
  op.SetAttr("k", k);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {795};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

