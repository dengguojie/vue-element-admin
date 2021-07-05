#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class matmulv2_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "matmulv2_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matmulv2_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(matmulv2_fusion_pass_test, matmulv2_fusion_pass_test_1) {
  ge::Graph graph("matmulv2_fusion_pass_test_1");

  // create matmulv2
  auto data_x1 = op::Data("data_x1");
  std::vector<int64_t> nd_shape_x1{16, 32};
  ge::TensorDesc desc_x1(ge::Shape(nd_shape_x1), FORMAT_ND, DT_FLOAT16);
  data_x1.update_input_desc_x(desc_x1);
  data_x1.update_output_desc_y(desc_x1);

  auto data_x2 = op::Data("data_x2");
  std::vector<int64_t> nd_shape_x2{32 ,16};
  ge::TensorDesc desc_x2(ge::Shape(nd_shape_x2), FORMAT_ND, DT_FLOAT16);
  data_x2.update_input_desc_x(desc_x2);
  data_x2.update_output_desc_y(desc_x2);

  auto matmulv2 = op::MatMulV2("MatMulV2")
                      .set_input_x1(data_x1)
                      .set_input_x2(data_x2)
                      .set_attr_transpose_x2(true);
  matmulv2.update_input_desc_x1(desc_x1);
  matmulv2.update_input_desc_x2(desc_x2);
  std::vector<int64_t> nchw_shape_y{16, 16};
  ge::TensorDesc desc_y(ge::Shape(nchw_shape_y), FORMAT_NCHW, DT_INT32);
  matmulv2.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{data_x1, data_x2};
  std::vector<Operator> outputs{matmulv2};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("MatMulV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findNode = false;
  EXPECT_EQ(findNode, false);
}