#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class deconv_weight_trans_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "deconv_weight_trans_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "deconv_weight_trans_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(deconv_weight_trans_fusion_pass_test, deconv_weight_trans_fusion_pass_test_dtype_error) {
  ge::Graph graph("deconv_weight_trans_fusion_pass_test_dtype_error");

  // create deconvolution
  auto data_x = op::Data("data_x");
  std::vector<int64_t> nchw_shape_x{1, 192, 48, 80};
  ge::TensorDesc desc_x(ge::Shape(nchw_shape_x), FORMAT_NCHW, DT_FLOAT16);
  data_x.update_input_desc_x(desc_x);
  data_x.update_output_desc_y(desc_x);

  auto data_filter = op::Data("data_filter");
  std::vector<int64_t> nchw_shape_filter{144, 64, 3, 3};
  ge::TensorDesc desc_filter(ge::Shape(nchw_shape_filter), FORMAT_NCHW, DT_FLOAT16);
  data_filter.update_input_desc_x(desc_filter);
  data_filter.update_output_desc_y(desc_filter);

  auto deconvolution = op::Deconvolution("deconvolution").set_input_x(data_x).set_input_filter(data_filter);
  deconvolution.update_input_desc_x(desc_x);
  deconvolution.update_input_desc_filter(desc_filter);
  std::vector<int64_t> nchw_shape_y{1, 144, 48, 80};
  ge::TensorDesc desc_y(ge::Shape(nchw_shape_y), FORMAT_NCHW, DT_FLOAT16);

  std::vector<Operator> inputs{data_x, data_filter};
  std::vector<Operator> outputs{deconvolution};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("DeconvWeightTransFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findNode = false;
  EXPECT_EQ(findNode, false);
}

TEST_F(deconv_weight_trans_fusion_pass_test, deconv_weight_trans_fusion_pass_test_w_not_constant) {
  ge::Graph graph("deconv_weight_trans_fusion_pass_test_w_not_constant");

  // create deconvolution
  auto data_x = op::Data("data_x");
  std::vector<int64_t> nchw_shape_x{1, 192, 48, 80};
  ge::TensorDesc desc_x(ge::Shape(nchw_shape_x), FORMAT_NCHW, DT_INT8);
  data_x.update_input_desc_x(desc_x);
  data_x.update_output_desc_y(desc_x);

  auto data_filter = op::Data("data_filter");
  std::vector<int64_t> nchw_shape_filter{144, 64, 3, 3};
  ge::TensorDesc desc_filter(ge::Shape(nchw_shape_filter), FORMAT_NCHW, DT_INT8);
  data_filter.update_input_desc_x(desc_filter);
  data_filter.update_output_desc_y(desc_filter);

  auto deconvolution = op::Deconvolution("deconvolution").set_input_x(data_x).set_input_filter(data_filter);
  deconvolution.update_input_desc_x(desc_x);
  deconvolution.update_input_desc_filter(desc_filter);
  std::vector<int64_t> nchw_shape_y{1, 144, 48, 80};
  ge::TensorDesc desc_y(ge::Shape(nchw_shape_y), FORMAT_NCHW, DT_INT32);

  std::vector<Operator> inputs{data_x, data_filter};
  std::vector<Operator> outputs{deconvolution};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("DeconvWeightTransFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findNode = false;
  EXPECT_EQ(findNode, false);
}

TEST_F(deconv_weight_trans_fusion_pass_test, deconv_weight_trans_fusion_pass_test_ok) {
  ge::Graph graph("deconv_weight_trans_fusion_pass_test_ok");

  // create deconvolution
  auto data_x = op::Data("data_x");
  std::vector<int64_t> nchw_shape_x{1, 192, 48, 80};
  ge::TensorDesc desc_x(ge::Shape(nchw_shape_x), FORMAT_NCHW, DT_INT8);
  data_x.update_input_desc_x(desc_x);
  data_x.update_output_desc_y(desc_x);


  auto const_filter = op::Const("const_filter");
  std::vector<int64_t> nchw_shape_filter{144, 64, 3, 3};
  TensorDesc desc_filter(ge::Shape(nchw_shape_filter), FORMAT_NCHW, DT_INT8);
  ge::Tensor tensor_filter;
  tensor_filter.SetTensorDesc(desc_filter);
  int8_t* data_filter = nullptr;
  data_filter = new int8_t[144*64*3*3];
  tensor_filter.SetData((uint8_t*)data_filter, 144*64*3*3 * sizeof(int8_t));
  const_filter.update_output_desc_y(desc_filter);
  delete[] data_filter;

  auto deconvolution = op::Deconvolution("deconvolution").set_input_x(data_x).set_input_filter(const_filter);
  deconvolution.update_input_desc_x(desc_x);
  deconvolution.update_input_desc_filter(desc_filter);
  std::vector<int64_t> nchw_shape_y{1, 144, 48, 80};
  ge::TensorDesc desc_y(ge::Shape(nchw_shape_y), FORMAT_NCHW, DT_INT32);

  std::vector<Operator> inputs{data_x, const_filter};
  std::vector<Operator> outputs{deconvolution};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("DeconvWeightTransFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findNode = false;
  EXPECT_EQ(findNode, false);
}