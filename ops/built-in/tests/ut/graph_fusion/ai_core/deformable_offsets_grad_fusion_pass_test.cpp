#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class deformable_offsets_grad_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "deformable_offsets_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "deformable_offsets_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(deformable_offsets_grad_fusion_pass_test, deformable_offsets_grad_fusion_pass_test_1) {
  ge::Graph graph("deformable_offsets_fusion_pass_test_1");

  auto grad_data = op::Data("grad_data");
  std::vector<int64_t> dims_grad{1, 304, 304, 256};
  ge::Shape shape_grad(dims_grad);
  ge::TensorDesc tensorDescGrad(shape_grad, FORMAT_NHWC, DT_FLOAT);
  grad_data.update_input_desc_x(tensorDescGrad);
  grad_data.update_output_desc_y(tensorDescGrad);

  // input data x
  auto x_data = op::Data("x_data");
  std::vector<int64_t> dims_x{1, 304, 304, 256};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
  x_data.update_input_desc_x(tensorDescX);
  x_data.update_output_desc_y(tensorDescX);

  // input data offset
  auto offset_data = op::Data("offset_data");
  std::vector<int64_t> dims_offset{1, 152, 152, 27};
  ge::Shape shape_offset(dims_offset);
  ge::TensorDesc tensorDescOffset(shape_offset, FORMAT_NHWC, DT_FLOAT);
  offset_data.update_input_desc_x(tensorDescOffset);
  offset_data.update_output_desc_y(tensorDescOffset);

  // get input data
  auto deformablegrad_op = op::DeformableOffsetsGrad("DeformableOffsetsGrad");
  deformablegrad_op.set_input_grad(grad_data);
  deformablegrad_op.set_input_x(x_data);
  deformablegrad_op.set_input_offsets(offset_data);

  // create attr
  std::vector<int64_t> strides_attr = {1, 2, 2, 1};
  std::vector<int64_t> pads_attr = {1, 1, 1, 1};
  std::vector<int64_t> ksize_attr = {3, 3};
  std::vector<int64_t> dilations_attr = {1, 1, 1, 1};
  deformablegrad_op.SetAttr("strides", strides_attr);
  deformablegrad_op.SetAttr("pads", pads_attr);
  deformablegrad_op.SetAttr("ksize", ksize_attr);
  deformablegrad_op.SetAttr("dilations", dilations_attr);
  deformablegrad_op.SetAttr("data_format", "NHWC");
  deformablegrad_op.SetAttr("deformable_groups", 1);
  deformablegrad_op.SetAttr("modulated", true);

  //  fussion pass
  std::vector<Operator> inputs{grad_data, x_data, offset_data};
  std::vector<Operator> outputs{deformablegrad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DeformableOffsetsGradFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findDeformableOffsets = false;

  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "DeformableOffsetsGrad") {
      findDeformableOffsets = true;
      break;
    }
  }
  EXPECT_EQ(findDeformableOffsets, true);
}
