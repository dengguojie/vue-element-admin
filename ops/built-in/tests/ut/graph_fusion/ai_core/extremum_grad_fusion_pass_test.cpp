//
// Created by c30002892 on 2020/9/5.
//

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "selection_ops.h"

using namespace ge;
using namespace op;

class extremum_grad_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(extremum_grad_fusion_pass_test, extremum_grad_fusion_pass_test_1) {
  ge::Graph graph("extremum_grad_fusion_pass_test_1");
  auto diag_input_data = op::Data("input_data");
  std::vector<int64_t> dims{3, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape);
  diag_input_data.update_input_desc_x(tensorDesc);
  diag_input_data.update_output_desc_y(tensorDesc);

  auto diag_input_data_2 = op::Data("input_data_2");
  std::vector<int64_t> dims_2{3, 32};
  ge::Shape shape_2(dims_2);
  ge::TensorDesc tensorDesc2(shape_2);
  diag_input_data_2.update_input_desc_x(tensorDesc2);
  diag_input_data_2.update_output_desc_y(tensorDesc2);

  auto diag_op = op::GreaterEqual("data1/Maximum_grad/data2");
  diag_op.set_input_x1(diag_input_data);
  diag_op.set_input_x2(diag_input_data_2);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(diag_op);
  std::vector<Operator> inputs{diag_input_data};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ExtremumGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool findDiagD = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{3, 32, 3, 32};
  for (auto node: compute_graph_ptr->GetAllNodes()) {

    if (node->GetType() == "DiagD") {
      findDiagD = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
//  EXPECT_EQ(findDiagD, true);
//  EXPECT_EQ(shapeMatch, true);
}

TEST_F(extremum_grad_fusion_pass_test, extremum_grad_fusion_pass_test_2) {
ge::Graph graph("extremum_grad_fusion_pass_test_2");
auto diag_input_data = op::Data("input_data");
std::vector<int64_t> dims{3, 32};
ge::Shape shape(dims);
ge::TensorDesc tensorDesc(shape);
diag_input_data.update_input_desc_x(tensorDesc);
diag_input_data.update_output_desc_y(tensorDesc);

auto diag_input_data_2 = op::Data("input_data_2");
std::vector<int64_t> dims_2{3, 32};
ge::Shape shape_2(dims_2);
ge::TensorDesc tensorDesc2(shape_2);
diag_input_data_2.update_input_desc_x(tensorDesc2);
diag_input_data_2.update_output_desc_y(tensorDesc2);

ge::Tensor const_tensor;
int64_t const_size = 1;
std::vector<int64_t> const_vec{const_size};
ge::Shape const_shape(const_vec);
ge::TensorDesc const_desc(const_shape, FORMAT_ND, DT_FLOAT);
const_desc.SetSize(const_size * sizeof(float));
const_tensor.SetTensorDesc(const_desc);
float* const_data = nullptr;
const_data = new float[const_size];
for (int i=0; i<const_size; i++) {
*(const_data + i) = 1;
}
const_tensor.SetData((uint8_t*)const_data, const_size * sizeof(float));

auto const1_const_op = op::Constant().set_attr_value(const_tensor);
const1_const_op.update_output_desc_y(const_desc);

ge::Tensor const_dz_tensor;
const_dz_tensor.SetTensorDesc(const_desc);
const_dz_tensor.SetData((uint8_t*)const_data, const_size * sizeof(float));
delete [] const_data;

auto constdz_const_op = op::Constant().set_attr_value(const_tensor);
constdz_const_op.update_output_desc_y(const_desc);

auto diag_op = op::GreaterEqual("data1/Maximum_grad/data2");
diag_op.set_input_x1(diag_input_data);
diag_op.set_input_x2(diag_input_data_2);

auto select0 = op::Select("data1/Maximum_grad/Select0")
    .set_input_condition(diag_op)
    .set_input_x1(const1_const_op)
    .set_input_x2(constdz_const_op);

auto end_op = op::Square("end_op_0");
end_op.set_input_x(select0);
std::vector<Operator> inputs{diag_input_data, diag_input_data_2};
std::vector<Operator> outputs{end_op};
graph.SetInputs(inputs).SetOutputs(outputs);
ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
fe::FusionPassTestUtils::RunGraphFusionPass("ExtremumGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
bool findSelect = false;
for (auto node: compute_graph_ptr->GetAllNodes()) {
if (node->GetType() == "Select") {
findSelect = true;
}
}
EXPECT_EQ(findSelect, true);
//  EXPECT_EQ(shapeMatch, true);
}