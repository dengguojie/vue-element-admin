#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class NonZeroProtoUT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonZeroProtoUT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonZeroProtoUT TearDown" << std::endl;
  }
};

TEST_F(NonZeroProtoUT, nonzero_test_1) {
  ge::op::NonZero op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({-2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  op.UpdateInputDesc("x", tensor_desc);

  op.SetAttr("transpose", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(NonZeroProtoUT, nonzero_test_2) {
  ge::op::NonZero op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, 2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  op.UpdateInputDesc("x", tensor_desc);

  op.SetAttr("transpose", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {-1, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(NonZeroProtoUT, nonzero_test_3) {
  ge::op::NonZero op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, -1});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  op.UpdateInputDesc("x", tensor_desc);

  op.SetAttr("transpose", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {-1, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(NonZeroProtoUT, nonzero_test_4) {
  ge::op::NonZero op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, 2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  op.UpdateInputDesc("x", tensor_desc);

  op.SetAttr("transpose", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {2, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(NonZeroProtoUT, nonzero_test_5) {
  ge::op::NonZero op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, -1});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  op.UpdateInputDesc("x", tensor_desc);

  op.SetAttr("transpose", true);
  ge::DataType dtype = ge::DT_INT32;
  op.SetAttr("dtype", dtype);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {2, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
