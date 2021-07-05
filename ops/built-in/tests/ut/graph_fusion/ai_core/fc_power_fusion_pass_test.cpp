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

class fc_power_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "fc_power_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "fc_power_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(fc_power_fusion_pass_test, fc_power_fusion_pass_test_1) {
    ge::Graph graph("fc_power_fusion_pass_test_1");
     auto input_data = op::Data("input_data");
    std::vector<int64_t> data_dims{2, 1, 32, 32};
    ge::Shape data_shape(data_dims);
    ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_INT8);
    input_data.update_input_desc_x(dataInputTensorDesc);
    input_data.update_output_desc_y(dataInputTensorDesc);

    auto fc_op = op::FullyConnection("fc");
    auto x_input_data = op::Data("fc_x_input_data");
    auto w_input_data = op::Data("fc_w_input_data");
    std::vector<int64_t> feature_dims{2, 1, 32, 32};
    std::vector<int64_t> weight_dims{2, 5, 32, 32};
    ge::Shape feature_shape(feature_dims);
    ge::Shape weight_shape(weight_dims);
    ge::TensorDesc featurenputTensorDesc(feature_shape, FORMAT_FRACTAL_NZ, DT_INT8);
    ge::TensorDesc weghtInputTensorDesc(weight_shape, FORMAT_FRACTAL_NZ, DT_INT8);
    x_input_data.update_input_desc_x(featurenputTensorDesc);
    x_input_data.update_output_desc_y(featurenputTensorDesc);
    w_input_data.update_input_desc_x(weghtInputTensorDesc);
    w_input_data.update_output_desc_y(weghtInputTensorDesc);


    auto power_op = op::Power("power");
    std::vector<int64_t> power_dims{2, 1, 32, 32};
    ge::Shape power_shape(power_dims);
    ge::TensorDesc ExpTensorDesc(power_shape, FORMAT_NC1HWC0, DT_INT8);
    power_op.update_input_desc_x(ExpTensorDesc);
    power_op.update_output_desc_y(ExpTensorDesc);
    float power = 2;
    float scale = 3;
    float shift = 4;
    power_op.set_attr_power(power)
            .set_attr_scale(scale)
            .set_attr_shift(shift);

    fc_op.set_input_x(input_data);
    power_op.set_input_x(fc_op);


    std::vector<Operator> inputs{input_data};
    std::vector<Operator> outputs{power_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("FullyConnectionPowerPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findNode = false;
    EXPECT_EQ(findNode, false);
}

TEST_F(fc_power_fusion_pass_test, fc_power_fusion_pass_test_2) {
    ge::Graph graph("fc_power_fusion_pass_test_2");
     auto input_data = op::Data("input_data");
    std::vector<int64_t> data_dims{2, 1, 16, 16};
    ge::Shape data_shape(data_dims);
    ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_data.update_input_desc_x(dataInputTensorDesc);
    input_data.update_output_desc_y(dataInputTensorDesc);

    auto fc_op = op::FullyConnection("fc");
    auto x_input_data = op::Data("fc_x_input_data");
    auto w_input_data = op::Data("fc_w_input_data");
    std::vector<int64_t> feature_dims{2, 1, 16, 16};
    std::vector<int64_t> weight_dims{2, 5, 16, 16};
    ge::Shape feature_shape(feature_dims);
    ge::Shape weight_shape(weight_dims);
    ge::TensorDesc featurenputTensorDesc(feature_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    ge::TensorDesc weghtInputTensorDesc(weight_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    x_input_data.update_input_desc_x(featurenputTensorDesc);
    x_input_data.update_output_desc_y(featurenputTensorDesc);
    w_input_data.update_input_desc_x(weghtInputTensorDesc);
    w_input_data.update_output_desc_y(weghtInputTensorDesc);


    auto power_op = op::Power("power");
    std::vector<int64_t> power_dims{2, 1, 16, 16};
    ge::Shape power_shape(power_dims);
    ge::TensorDesc ExpTensorDesc(power_shape, FORMAT_NC1HWC0, DT_FLOAT16);
    power_op.update_input_desc_x(ExpTensorDesc);
    power_op.update_output_desc_y(ExpTensorDesc);
    float power = 2;
    float scale = 3;
    float shift = 4;
    power_op.set_attr_power(power)
            .set_attr_scale(scale)
            .set_attr_shift(shift);

    fc_op.set_input_x(input_data);
    power_op.set_input_x(fc_op);


    std::vector<Operator> inputs{input_data};
    std::vector<Operator> outputs{power_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("FullyConnectionPowerPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findNode = false;
    EXPECT_EQ(findNode, false);
}

TEST_F(fc_power_fusion_pass_test, fc_power_fusion_pass_test_3) {
  ge::Graph graph("fc_power_fusion_pass_test_3");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> data_dims{2, 1, 16, 16};
  ge::Shape data_shape(data_dims);
  ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  input_data.update_input_desc_x(dataInputTensorDesc);
  input_data.update_output_desc_y(dataInputTensorDesc);

  auto fc_op = op::FullyConnection("fc");
  auto x_input_data = op::Data("fc_x_input_data");
  auto w_input_data = op::Data("fc_w_input_data");
  std::vector<int64_t> feature_dims{2, 1, 16, 16};
  std::vector<int64_t> weight_dims{2, 5, 16, 16};
  ge::Shape feature_shape(feature_dims);
  ge::Shape weight_shape(weight_dims);
  ge::TensorDesc featurenputTensorDesc(feature_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  ge::TensorDesc weghtInputTensorDesc(weight_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  x_input_data.update_input_desc_x(featurenputTensorDesc);
  x_input_data.update_output_desc_y(featurenputTensorDesc);
  w_input_data.update_input_desc_x(weghtInputTensorDesc);
  w_input_data.update_output_desc_y(weghtInputTensorDesc);

  auto power_op = op::Power("power");
  std::vector<int64_t> power_dims{2, 1, 16, 16};
  ge::Shape power_shape(power_dims);
  ge::TensorDesc ExpTensorDesc(power_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  power_op.update_input_desc_x(ExpTensorDesc);
  power_op.update_output_desc_y(ExpTensorDesc);
  float power = 1;
  float scale = 1;
  float shift = 4;
  power_op.set_attr_power(power).set_attr_scale(scale).set_attr_shift(shift);

  fc_op.set_input_x(input_data);
  power_op.set_input_x(fc_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{power_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("FullyConnectionPowerPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findNode = false;
  EXPECT_EQ(findNode, false);
}

TEST_F(fc_power_fusion_pass_test, fc_power_fusion_pass_test_4) {
  ge::Graph graph("fc_power_fusion_pass_test_4");

  // create fullyconnection
  auto data_x = op::Data("data_x");
  std::vector<int64_t> nchw_shape_x{2, 1, 4, 4};
  ge::TensorDesc desc_x(ge::Shape(nchw_shape_x), FORMAT_NCHW, DT_FLOAT16);
  data_x.update_input_desc_x(desc_x);
  data_x.update_output_desc_y(desc_x);

  auto data_w = op::Data("data_w");
  std::vector<int64_t> nchw_shape_w{16, 2, 4, 4};
  ge::TensorDesc desc_w(ge::Shape(nchw_shape_w), FORMAT_NCHW, DT_FLOAT16);
  data_w.update_input_desc_x(desc_w);
  data_w.update_output_desc_y(desc_w);
 
  auto data_b = op::Data("data_b");
  std::vector<int64_t> nd_shape_b{16};
  ge::TensorDesc desc_b(ge::Shape(nd_shape_b), FORMAT_ND, DT_FLOAT16);
  data_b.update_input_desc_x(desc_b);
  data_b.update_output_desc_y(desc_b);

  auto fullyconnection = op::FullyConnection("fc")
                    .set_input_x(data_x)
                    .set_input_w(data_w)
                    .set_input_b(data_b);
  fullyconnection.update_input_desc_x(desc_x);
  fullyconnection.update_input_desc_w(desc_w);
  fullyconnection.update_input_desc_b(desc_b);
  std::vector<int64_t> fractal_nz_shape_y{2, 16, 1, 1};
  ge::TensorDesc out_desc_fc(ge::Shape(fractal_nz_shape_y), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  fullyconnection.update_output_desc_y(out_desc_fc);

  // create power
  auto power_op = op::Power("power");
  std::vector<int64_t> power_dims{2, 1, 16, 16};
  ge::Shape power_shape(power_dims);
  ge::TensorDesc ExpTensorDesc(power_shape, FORMAT_NC1HWC0, DT_FLOAT16);
  power_op.update_input_desc_x(ExpTensorDesc);
  power_op.update_output_desc_y(ExpTensorDesc);
  float power = 1;
  float scale = 1;
  float shift = 4;
  power_op.set_attr_power(power).set_attr_scale(scale).set_attr_shift(shift);
  power_op.set_input_x(fullyconnection);

  std::vector<Operator> inputs{data_x};
  std::vector<Operator> outputs{power_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("FullyConnectionPowerPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findNode = false;
  EXPECT_EQ(findNode, false);
}
