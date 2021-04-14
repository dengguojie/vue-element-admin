#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nn_norm_ops.h"
#include "nn_detect_ops.h"
#include "fp16_t.hpp"
#include "reduce_ops.h"
using namespace ge;
using namespace op;

class roialign_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "roialign_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "roialign_fusion_test TearDown" << std::endl;
  }
};

TEST_F(roialign_fusion_test, roialign_fusion_test_1) {
  ge::Graph graph("roialign_fusion_test_1");
  auto x = op::Data("x");
  auto x1 = op::Data("x1");
  auto x2 = op::Data("x2");

  std::vector<int64_t> dims{1, 1, 10, 10};
  ge::Shape shape(dims);
  std::vector<int64_t> dims1{3, 4};
  ge::Shape shape1(dims1);
  std::vector<int64_t> dims2{3};
  ge::Shape shape2(dims2);
  std::vector<int64_t> dims3{3, 1, 5, 5};
  ge::Shape shape3(dims3);

  ge::TensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT64);
  ge::TensorDesc tensor_desc3(shape3, ge::FORMAT_NCHW, ge::DT_FLOAT);

  x.update_input_desc_x(tensor_desc);
  x.update_output_desc_y(tensor_desc);
  x1.update_input_desc_x(tensor_desc1);
  x1.update_output_desc_y(tensor_desc1);
  x2.update_input_desc_x(tensor_desc2);
  x2.update_output_desc_y(tensor_desc2);

  auto roialign_op = op::ROIAlign("ROIAlign");
  roialign_op.set_input_features(x).set_input_rois(x1).set_input_rois_n(x2);
  roialign_op.set_attr_spatial_scale(1).set_attr_pooled_height(5).set_attr_pooled_width(5).set_attr_sample_num(2);
  auto desc = roialign_op.GetInputDesc(1);
  int dim_num = desc.GetShape().GetDimNum();
  std::cout << "roialign_fusion_test DimNum" << dim_num << std::endl;
  std::vector<Operator> inputs{x, x1, x2};
  std::vector<Operator> outputs{roialign_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ROIAlignFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool is_have_cast = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatD") {
      is_have_cast = true;
    }
  }
  EXPECT_EQ(is_have_cast, true);
}

TEST_F(roialign_fusion_test, roialign_fusion_test_2) {
  ge::Graph graph("roialign_fusion_test_2");
  auto x = op::Data("x");
  auto x1 = op::Data("x1");
  auto x2 = op::Data("x2");

  std::vector<int64_t> dims{1, 1, 10, 10};
  ge::Shape shape(dims);
  std::vector<int64_t> dims1{3, 4, 3};
  ge::Shape shape1(dims1);
  std::vector<int64_t> dims2{3};
  ge::Shape shape2(dims2);
  std::vector<int64_t> dims3{3, 1, 5, 5};
  ge::Shape shape3(dims3);

  ge::TensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT64);
  ge::TensorDesc tensor_desc3(shape3, ge::FORMAT_NCHW, ge::DT_FLOAT);

  x.update_input_desc_x(tensor_desc);
  x.update_output_desc_y(tensor_desc);
  x1.update_input_desc_x(tensor_desc1);
  x1.update_output_desc_y(tensor_desc1);
  x2.update_input_desc_x(tensor_desc2);
  x2.update_output_desc_y(tensor_desc2); 

  auto roialign_op = op::ROIAlign("ROIAlign");
  roialign_op.set_input_features(x).set_input_rois(x1).set_input_rois_n(x2);
  roialign_op.set_attr_spatial_scale(1).set_attr_pooled_height(5).set_attr_pooled_width(5).set_attr_sample_num(2);
  auto desc = roialign_op.GetInputDesc(1);
  int dim_num = desc.GetShape().GetDimNum();
  std::cout << "roialign_fusion_test DimNum" << dim_num << std::endl;
  std::vector<Operator> inputs{x, x1, x2};
  std::vector<Operator> outputs{roialign_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ROIAlignFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool is_have_cast = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatD") {
      is_have_cast = true;
    }
  }
  EXPECT_EQ(is_have_cast, false);
}

TEST_F(roialign_fusion_test, roialign_fusion_test_3) {
  ge::Graph graph("roialign_fusion_test_3");
  auto x = op::Data("x");
  auto x1 = op::Data("x1");
  auto x2 = op::Data("x2");

  std::vector<int64_t> dims{1, 1, 10, 10};
  ge::Shape shape(dims);
  std::vector<int64_t> dims1{3, 4};
  ge::Shape shape1(dims1);
  std::vector<int64_t> dims2{4};
  ge::Shape shape2(dims2);
  std::vector<int64_t> dims3{3, 1, 5, 5};
  ge::Shape shape3(dims3);

  ge::TensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_INT64);
  ge::TensorDesc tensor_desc3(shape3, ge::FORMAT_NCHW, ge::DT_FLOAT);

  x.update_input_desc_x(tensor_desc);
  x.update_output_desc_y(tensor_desc);
  x1.update_input_desc_x(tensor_desc1);
  x1.update_output_desc_y(tensor_desc1);
  x2.update_input_desc_x(tensor_desc2);
  x2.update_output_desc_y(tensor_desc2);

  auto roialign_op = op::ROIAlign("ROIAlign");
  roialign_op.set_input_features(x).set_input_rois(x1).set_input_rois_n(x2);
  roialign_op.set_attr_spatial_scale(1).set_attr_pooled_height(5).set_attr_pooled_width(5).set_attr_sample_num(2);
  auto desc = roialign_op.GetInputDesc(1);
  int dim_num = desc.GetShape().GetDimNum();
  std::cout << "roialign_fusion_test DimNum" << dim_num << std::endl;
  std::vector<Operator> inputs{x, x1, x2};
  std::vector<Operator> outputs{roialign_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ROIAlignFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool is_have_cast = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatD") {
      is_have_cast = true;
    }
  }
  EXPECT_EQ(is_have_cast, false);
}