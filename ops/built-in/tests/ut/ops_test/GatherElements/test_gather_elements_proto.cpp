#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class GatherElementsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_elements SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_elements TearDown" << std::endl;
  }
};

TEST_F(GatherElementsTest, gather_elements_test_1) {
  ge::op::GatherElements op;
  ge::TensorDesc input;
  ge::Shape shape_input({2, 3, 4});
  input.SetDataType(ge::DT_FLOAT16);
  input.SetShape(shape_input);
  input.SetOriginShape(shape_input);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{2, 2}, {3, 3}, {4, 4}};
  input.SetShapeRange(input_range);
  op.UpdateInputDesc("x", input);

  ge::TensorDesc index;
  ge::Shape shape_index({2, 1, 4});
  index.SetDataType(ge::DT_INT64);
  index.SetShape(shape_index);
  index.SetOriginShape(shape_index);
  std::vector<std::pair<int64_t, int64_t>> index_range = {{2, 2}, {1, 1}, {4, 4}};
  index.SetShapeRange(index_range);
  op.UpdateInputDesc("index", index);

  int attr_value = 1;
  op.SetAttr("dim", attr_value);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 1, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{2, 2}, {1, 1}, {4, 4}};
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output_desc.GetShapeRange(output_range);
  EXPECT_EQ(output_range, expect_output_range);
}

TEST_F(GatherElementsTest, gather_elements_test_failed) {
  ge::op::GatherElements op;
  ge::TensorDesc input;
  ge::Shape shape_input({2, 3, 4});
  input.SetDataType(ge::DT_FLOAT16);
  input.SetShape(shape_input);
  input.SetOriginShape(shape_input);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{2, 2}, {3, 3}, {4, 4}};
  input.SetShapeRange(input_range);
  op.UpdateInputDesc("x", input);

  ge::TensorDesc index;
  ge::Shape shape_index({2, 1, 4});
  index.SetDataType(ge::DT_FLOAT16);
  index.SetShape(shape_index);
  index.SetOriginShape(shape_index);
  std::vector<std::pair<int64_t, int64_t>> index_range = {{2, 2}, {1, 1}, {4, 4}};
  index.SetShapeRange(input_range);
  op.UpdateInputDesc("index", index);

  int attr_value = 1;
  op.SetAttr("dim", attr_value);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GatherElementsTest, gather_elements_test_failed_dim) {
  ge::op::GatherElements op;
  ge::TensorDesc input;
  ge::Shape shape_input({2, 3, 4});
  input.SetDataType(ge::DT_FLOAT16);
  input.SetShape(shape_input);
  input.SetOriginShape(shape_input);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{2, 2}, {3, 3}, {4, 4}};
  input.SetShapeRange(input_range);
  op.UpdateInputDesc("x", input);

  ge::TensorDesc index;
  ge::Shape shape_index({2, 1});
  index.SetDataType(ge::DT_INT32);
  index.SetShape(shape_index);
  index.SetOriginShape(shape_index);
  std::vector<std::pair<int64_t, int64_t>> index_range = {{2, 2}, {1, 1}};
  index.SetShapeRange(input_range);
  op.UpdateInputDesc("index", index);

  int attr_value = 1;
  op.SetAttr("dim", attr_value);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GatherElementsTest, gather_elements_test_4) {
  ge::op::GatherElements op;
  ge::TensorDesc input;
  ge::Shape shape_input({-1, -1});
  input.SetDataType(ge::DT_FLOAT16);
  input.SetShape(shape_input);
  input.SetOriginShape(shape_input);
  std::vector<std::pair<int64_t, int64_t>> input_range = {{64, 64}, {64, 64}};
  input.SetShapeRange(input_range);
  op.UpdateInputDesc("x", input);

  ge::TensorDesc index;
  ge::Shape shape_index({-1, -1});
  index.SetDataType(ge::DT_INT64);
  index.SetShape(shape_index);
  index.SetOriginShape(shape_index);
  std::vector<std::pair<int64_t, int64_t>> index_range = {{64, 64}, {64, 64}};
  index.SetShapeRange(index_range);
  op.UpdateInputDesc("index", index);

  int attr_value = 1;
  op.SetAttr("dim", attr_value);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expect_output_range = {{64, 64}, {64, 64}};
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output_desc.GetShapeRange(output_range);
  EXPECT_EQ(output_range, expect_output_range);
}