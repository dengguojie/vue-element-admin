#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "framework/common/types.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "reduce_ops.h"

using namespace ge;
using namespace op;

class batch_matmul_reduce_mean_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "batch_matmul_reduce_mean_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batch_matmul_reduce_mean_fusion_pass_test TearDown" << std::endl;
  }
};

bool ExecuteCorrectly(ge::ComputeGraphPtr compute_graph_ptr) {
  int cnt = 0;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadD" || node->GetType() == "SliceD") {
      cnt++;
    }
  }
  bool res = (cnt == 3) ? true : false;
  return res;
}

TEST_F(batch_matmul_reduce_mean_fusion_pass_test, batchmatmul_pattern_1) {
  std::cout << "enter batch_matmul_reduce_mean_fusion_pass_test.batchmatmul_pattern_1" << std::endl;
  ge::Graph graph("batchmatmul_pattern_1");
  ge::Shape x1_shape({1000, 48, 324});
  ge::Shape x2_shape({324, 10});
  ge::Shape add_const_shape({10});

  ge::TensorDesc x1_desc(x1_shape, FORMAT_ND, DT_FLOAT16);
  x1_desc.SetOriginShape(x1_shape);
  auto data_x1 = op::Data("data_x1");
  data_x1.update_input_desc_x(x1_desc);
  data_x1.update_output_desc_y(x1_desc);

  ge::TensorDesc x2_desc(x2_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor x2_tensor(x2_desc);
  auto const_x2 = op::Const().set_attr_value(x2_tensor);

  auto batchmatmul_op = op::BatchMatMulV2("batchmatmul_op")
    .set_input_x1(data_x1)
    .set_input_x2(const_x2)
    .set_attr_adj_x1(false)
    .set_attr_adj_x2(false);

  ge::TensorDesc add_const_desc(add_const_shape, FORMAT_ND, DT_FLOAT16);
  ge::Tensor add_const_tensor(add_const_desc);
  auto add_const = op::Const().set_attr_value(add_const_tensor);

  auto add_op = op::Add("add_op")
    .set_input_x1(batchmatmul_op)
    .set_input_x2(add_const);

  auto relu_op = op::Relu("relu_op")
    .set_input_x(add_op);

  std::vector<int64_t> axes;
  axes.push_back(1);

  auto reduce_mean_op = op::ReduceMeanD("reduce_mean_op")
    .set_input_x(relu_op)
    .set_attr_axes(axes)
    .set_attr_keep_dims(false);

  auto end_op = op::Square("end_op");
  end_op.set_input_x(reduce_mean_op);

  std::vector<Operator> inputs{data_x1, const_x2, add_const};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchMatMulReduceMeanFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  EXPECT_EQ(ExecuteCorrectly(compute_graph_ptr), true);
}
