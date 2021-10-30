#include <gtest/gtest.h>
#include <iostream>
#include <climits>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"

class NonZeroWithValueProtoUT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonZeroWithValueProtoUT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonZeroWithValueProtoUT TearDown" << std::endl;
  }
};

TEST_F(NonZeroWithValueProtoUT, nonzerowithvalue_test_1) {
  ge::op::NonZeroWithValue op;
  ge::TensorDesc tensor_desc;
  ge::Shape shape({-2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("transpose", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("index");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(NonZeroWithValueProtoUT, nonzerowithvalue_test_2) {
  ge::op::NonZeroWithValue op;
  ge::TensorDesc tensor_desc;
  ge::Shape shape({2, 2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("transpose", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_count = op.GetOutputDesc("count");
  EXPECT_EQ(output_count.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {1};
  EXPECT_EQ(output_count.GetShape().GetDims(), expected_output_shape);

  auto output_value = op.GetOutputDesc("value");
  EXPECT_EQ(output_value.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_value = {4};
  EXPECT_EQ(output_value.GetShape().GetDims(), expected_output_shape_value);

  auto output_index = op.GetOutputDesc("index");
  EXPECT_EQ(output_index.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape_index = {8};
  EXPECT_EQ(output_index.GetShape().GetDims(), expected_output_shape_index);
}