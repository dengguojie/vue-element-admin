#include <algorithm>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "nonlinear_fuc_ops.h"
#include "nn_norm_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class group_norm_squeeze_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "group_norm_squeeze_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "group_norm_squeeze_fusion_test TearDown" << std::endl;
  }
};

TEST_F(group_norm_squeeze_fusion_test, group_norm_squeeze_fusion_test1) {
  ge::Graph graph("group_norm_squeeze_fusion_test");

  std::vector<float> const_v_scales = {2.0, 1.0, 1.0, 1.0};
  std::vector<ge::op::Const> const_op_s = {};
  for (auto const_v_scale : const_v_scales) {
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(&const_v_scale), sizeof(float));
    auto const_op = ge::op::Const().set_attr_value(tensor);
    const_op_s.emplace_back(const_op);
  }

  auto shape0_shape = ge::Shape({3});
  TensorDesc desc_input_size0(shape0_shape, FORMAT_ND, DT_INT64);
  Tensor shape0_tensor(desc_input_size0);
  uint64_t *shape0_tensor_value = new uint64_t[3]{8, 512, 203};
  shape0_tensor.SetData((uint8_t *) shape0_tensor_value, 3 * sizeof(uint64_t));
  auto shape0 = op::Constant().set_attr_value(shape0_tensor);
  delete []shape0_tensor_value;

  auto shape1_shape = ge::Shape({4});
  TensorDesc desc_input_size1(shape1_shape, FORMAT_ND, DT_INT64);
  Tensor shape1_tensor(desc_input_size1);
  uint64_t *shape1_tensor_value = new uint64_t[4]{8, 512, 1, 203};
  shape1_tensor.SetData((uint8_t *) shape1_tensor_value, 4 * sizeof(uint64_t));
  auto shape1 = op::Constant().set_attr_value(shape1_tensor);
  delete []shape1_tensor_value;

  ge::Operator::OpListInt axis = {2};
  auto data0 = op::Data();
  auto squeeze0 = op::Squeeze("Squeeze").set_input_x(data0).set_attr_axis(axis);
  auto instancenorm = op::InstanceNorm("InstanceNorm").set_input_x(squeeze0).set_input_gamma(const_op_s[0]).set_input_beta(const_op_s[1])
                      .set_attr_data_format("NCHW").set_attr_epsilon(0.006);
  auto mul = op::Mul("Mul").set_input_x1(instancenorm, "y").set_input_x2(const_op_s[2]);
  auto add = op::Add("Add").set_input_x1(mul).set_input_x2(const_op_s[3]);
  ge::TensorDesc data0_desc(ge::Shape({8, 512, 1, 203}), FORMAT_NCHW,  DT_FLOAT);
  ge::TensorDesc squeeze_y_desc(ge::Shape({8, 512, 203}), FORMAT_ND,  DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  squeeze0.update_output_desc_y(squeeze_y_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{add};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("GroupNormSqueezeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch = false;
  vector<int64_t> expectShape{8, 512, 1, 203};
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "GroupNorm") {
      findOp = true;
      auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
      std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
      if (dims == expectShape) {
        shapeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
}
