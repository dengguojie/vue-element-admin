#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class tile_fusion_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "tile_fusion_pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tile_fusion_pass TearDown" << std::endl;
  }
};

TEST_F(tile_fusion_pass_test, tile_fusion_pass_test_1) {
  ge::Graph graph("tile_fusion_pass_test_1");
  auto shape_data = vector<int64_t>({1, 1, 1, 1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT16);
  
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[4]{2, 2, 2, 2};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, 4 * sizeof(uint32_t));
  
  auto tile_multiples = op::Const("multiples")
          .set_attr_value(multiples_tensor);
  // tiled op
  auto tile = op::Tile("tile");
  tile.set_input_x(data);
  tile.set_input_multiples(tile_multiples);
  
  TensorDesc tile_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc tile_input_desc_multiples(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc tile_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  tile.update_input_desc_x(tile_input_desc_x);
  tile.update_input_desc_multiples(tile_input_desc_multiples);
  tile.update_output_desc_y(tile_output_desc_y);
  
  std::vector<Operator> inputs{data};
  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(tile);
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("TileConstToAttrFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");
  
  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TileD") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, true);
  delete[] multiples_tensor_value;

}

TEST_F(tile_fusion_pass_test, tile_fusion_pass_test_2) {
  ge::Graph graph("tile_fusion_pass_test_2");
  auto shape_data = vector<int64_t>({1, 1, 1, 1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT16);
  
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[4]{1, 1, 1, 1};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, 4 * sizeof(uint32_t));
  
  auto tile_multiples = op::Const("multiples")
          .set_attr_value(multiples_tensor);
  // tiled op
  auto tile = op::Tile("tile");
  tile.set_input_x(data);
  tile.set_input_multiples(tile_multiples);
  
  TensorDesc tile_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc tile_input_desc_multiples(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc tile_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  tile.update_input_desc_x(tile_input_desc_x);
  tile.update_input_desc_multiples(tile_input_desc_multiples);
  tile.update_output_desc_y(tile_output_desc_y);
  
  std::vector<Operator> inputs{data};
  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(tile);
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("TileConstToAttrFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");
  
  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TileD") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, false);
  delete[] multiples_tensor_value;

}

TEST_F(tile_fusion_pass_test, tile_fusion_pass_test_3) {
  ge::Graph graph("tile_fusion_pass_test_3");
  auto shape_data = vector<int64_t>({1, 1, 1, 1});
  TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NCHW, DT_FLOAT16);
  
  auto data = op::Data("data");
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  
  auto multiples_shape = ge::Shape({4});
  TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT64);
  // TensorDesc desc_weight_1(weight_shape, FORMAT_NCHW, DT_FLOAT16);
  Tensor multiples_tensor(desc_input_size_1);
  uint32_t *multiples_tensor_value = new uint32_t[4]{2, 2, 2, 2};
  multiples_tensor.SetData((uint8_t *) multiples_tensor_value, 4 * sizeof(uint32_t));
  
  auto tile_multiples = op::Const("multiples")
          .set_attr_value(multiples_tensor);
  // tiled op
  auto tile = op::Tile("tile");
  tile.set_input_x(data);
  tile.set_input_multiples(tile_multiples);
  
  TensorDesc tile_input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc tile_input_desc_multiples(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc tile_output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
  tile.update_input_desc_x(tile_input_desc_x);
  tile.update_input_desc_multiples(tile_input_desc_multiples);
  tile.update_output_desc_y(tile_output_desc_y);
  
  std::vector<Operator> inputs{data};
  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(tile);
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("TileConstToAttrFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "diag_fusion_test_1_after");
  
  bool findD = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TileD") {
      findD = true;
    }
  }
  EXPECT_EQ(findD, true);
  delete[] multiples_tensor_value;

}