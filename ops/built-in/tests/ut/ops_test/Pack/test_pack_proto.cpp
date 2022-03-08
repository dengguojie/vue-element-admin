#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class pack_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "pack_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "pack_infer_test TearDown" << std::endl;
  }
};

TEST_F(pack_infer_test, pack_infer_test_1) {
  // input x1 shape {-1, -1}
  // input x1 range {1, -1} {1, -1}
  // input x2 shape {-1, -1}
  // input x2 range {1, -1} {1, -1}
  // axis attr = 0
  // output shape {2, -1, -1}
  // output shape {2, 2} {1, -1} {1, -1}
  ge::Graph graph("pack_infer_test_1");
  auto shape_x1 = vector<int64_t>({-1, -1});
  TensorDesc desc_data_x1(ge::Shape(shape_x1), FORMAT_NCHW, DT_FLOAT16);
  desc_data_x1.SetOriginShape(ge::Shape(shape_x1));
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  x1_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  desc_data_x1.SetShapeRange(x1_range);

  auto data_x1 = op::Data("x1");
  data_x1.update_input_desc_x(desc_data_x1);
  data_x1.update_output_desc_y(desc_data_x1);

  auto shape_x2 = vector<int64_t>({-1, -1});
  TensorDesc desc_data_x2(ge::Shape(shape_x2), FORMAT_NCHW, DT_FLOAT16);
  desc_data_x2.SetOriginShape(ge::Shape(shape_x2));
  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  x2_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  desc_data_x2.SetShapeRange(x2_range);
  
  auto data_x2 = op::Data("x2");
  data_x2.update_input_desc_x(desc_data_x2);
  data_x2.update_output_desc_y(desc_data_x2);

  // test op
  auto test_layer = op::Pack("Pack");
  test_layer.create_dynamic_input_x(2)
            .set_dynamic_input_x(0, data_x1)
            .set_dynamic_input_x(1, data_x2)
            .set_attr_axis(0)
            .set_attr_N(2);

  std::vector<Operator> inputs{data_x1, data_x2};
  std::vector<Operator> outputs{test_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  test_layer.InferShapeAndType();

  bool findOp = false;
  bool shapeMatch = false;
  bool shapeRangeMatch = false;

  // set expect_shape
  auto expect_shape = vector<int64_t>({2, -1, -1});
  // set expect_range
  std::vector<std::pair<int64_t, int64_t>> expect_range;
  expect_range.push_back(std::pair<int64_t, int64_t>{2, 2});
  expect_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  expect_range.push_back(std::pair<int64_t, int64_t>{1, -1});

  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Pack") {
      findOp = true;
      auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> output_shape = outputDesc.GetShape().GetDims();
      std::vector<std::pair<int64_t, int64_t>> output_range;
      outputDesc.GetShapeRange(output_range);
      if (output_shape == expect_shape) {
          shapeMatch = true;
      }
      if (output_range == expect_range) {
          shapeRangeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
  EXPECT_EQ(shapeRangeMatch, true);
}

TEST_F(pack_infer_test, pack_infer_test_2) {
  ge::op::Pack op;
  std::vector<std::pair<int64_t,int64_t>> shape_range;
  shape_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  shape_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N", 2);
  op.SetAttr("axis", 1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, 2, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range;
  expected_shape_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  expected_shape_range.push_back(std::pair<int64_t, int64_t>{2, 2});
  expected_shape_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(pack_infer_test, pack_infer_test_3) {
  ge::op::Pack op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {
      {3, 3},
      {2, 5},
      {2, 5},
      {5, 5},
  };
  auto tensor_desc = create_desc_shape_range({3, -1, -1, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, -1, -1, 5},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);

  shape_range = {
      {3, 3},
      {1, 10},
      {3, 6},
      {1, 10},
  };
  tensor_desc = create_desc_shape_range({3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, -1, -1, -1}, ge::FORMAT_ND,
                                        shape_range);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("axis", 0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {3, 3, -1, -1, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {3, 3}, {3, 3}, {2, 5}, {3, 5}, {5, 5},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(pack_infer_test, pack_infer_test_4) {
  ge::op::Pack op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {
      {3, 3},
      {2, 5},
      {2, 5},
      {5, 5},
  };
  auto tensor_desc = create_desc_shape_range({3, -1, -1, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, -1, -1, 5},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_x(3);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);

  shape_range = {};
  tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, shape_range);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 2, tensor_desc);
  op.SetAttr("N", 3);
  op.SetAttr("axis", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {3, -1, 3, -1, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {3, 3}, {2, 5}, {3, 3}, {2, 5}, {5, 5},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(pack_infer_test, pack_infer_test_5) {
  ge::op::Pack op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, shape_range);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N", 2);
  op.SetAttr("axis", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(pack_infer_test, pack_infer_test_6) {
  ge::op::Pack op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::FORMAT_ND, shape_range);

  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N", 2);
  op.SetAttr("axis", 0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
