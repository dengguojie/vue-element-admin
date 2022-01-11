#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class dynamic_get_next : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_get_next SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_get_next TearDown" << std::endl;
  }
};

TEST_F(dynamic_get_next, dynamic_get_next_infershape_test_1) {
  ge::op::DynamicGetNext op;
  op.UpdateInputDesc("x", create_desc_with_ori({}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.create_dynamic_output_y(2);
  std::vector<ge::DataType> output_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("output_types", output_types);
  std::vector<int64_t> shape{-1,-1};
  ge::Operator::OpListListInt output_shapes{shape, shape};
  op.SetAttr("output_shapes", output_shapes);
  std::string dynamic_graph_execute_mode = "lazy_recompile";
  op.SetAttr("_dynamic_graph_execute_mode", dynamic_graph_execute_mode);
  std::string getnext_inputs_shape_range = "";
  op.SetAttr("_getnext_inputs_shape_range", getnext_inputs_shape_range);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetDynamicOutputDesc("y", 0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1},{1,-1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(dynamic_get_next, dynamic_get_next_infershape_test_2) {
  ge::op::DynamicGetNext op;
  op.UpdateInputDesc("x", create_desc_with_ori({}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.create_dynamic_output_y(2);
  std::vector<ge::DataType> output_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("output_types", output_types);
  std::vector<int64_t> shape{-1,-1};
  ge::Operator::OpListListInt output_shapes{shape, shape};
  op.SetAttr("output_shapes", output_shapes);
  std::string dynamic_graph_execute_mode = "dynamic_execute";
  op.SetAttr("_dynamic_graph_execute_mode", dynamic_graph_execute_mode);
  std::string getnext_inputs_shape_range = "[128,3~5],[3~5,3~5]";
  op.SetAttr("_getnext_inputs_shape_range", getnext_inputs_shape_range);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetDynamicOutputDesc("y", 0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{128,128},{3,5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);

  auto output_desc2 = op.GetDynamicOutputDesc("y", 1);
  EXPECT_EQ(output_desc2.GetDataType(), ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range2 = {{3,5},{3,5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range2;
  output_desc2.GetShapeRange(output_shape_range2);
  EXPECT_EQ(output_shape_range2, expected_output_shape_range2);
}

TEST_F(dynamic_get_next, dynamic_get_next_infershape_test_3) {
  ge::op::DynamicGetNextV2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.create_dynamic_output_y(2);
  std::vector<ge::DataType> output_types{ ge::DT_INT32, ge::DT_INT32 };
  op.SetAttr("output_types", output_types);
  std::vector<int64_t> shape{-1,-1};
  ge::Operator::OpListListInt output_shapes{shape, shape};
  op.SetAttr("output_shapes", output_shapes);
  std::string dynamic_graph_execute_mode = "dynamic_execute";
  op.SetAttr("_dynamic_graph_execute_mode", dynamic_graph_execute_mode);
  std::string getnext_inputs_shape_range = "[128,3~5],[3~5,3~5]";
  op.SetAttr("_getnext_inputs_shape_range", getnext_inputs_shape_range);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetDynamicOutputDesc("y", 0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{128,128},{3,5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
  auto output_desc2 = op.GetDynamicOutputDesc("y", 1);
  EXPECT_EQ(output_desc2.GetDataType(), ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range2 = {{3,5},{3,5}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range2;
  output_desc2.GetShapeRange(output_shape_range2);
  EXPECT_EQ(output_shape_range2, expected_output_shape_range2);
}
