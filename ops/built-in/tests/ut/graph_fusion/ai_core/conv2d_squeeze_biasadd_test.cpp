#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "state_ops.h"
#include "fusion_pass_test_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
using namespace op;

class conv2d_squeeze_biasadd_test : public testing::Test {
protected:
    static void SetUpTestCase() 
    {
        std::cout << "conv2d_squeeze_biasadd_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv2d_squeeze_biasadd_test TearDown" << std::endl;
    }
};
TEST_F(conv2d_squeeze_biasadd_test, conv2d_squeeze_biasadd_test_1) {
    ge::Graph graph("conv2d_squeeze_biasadd_test_1");
    auto conv_input_x_data = op::Data("conv_input_x_data");
    ge::TensorDesc tensorDescX(ge::Shape({1, 56, 56, 64}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    TensorDesc filter_desc(ge::Shape({64, 1, 1, 64}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    filter_desc.SetOriginFormat(FORMAT_NHWC);
    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    float * filter_value = new float[64 * 64];
    filter.SetTensorDesc(filter_desc);
    filter.SetData((uint8_t*)filter_value, 64 * 64 * 4);
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filter_desc);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
           .set_input_filter(filter_0)
           .set_attr_strides({1, 2, 2, 1})
           .set_attr_pads({1, 1, 1, 1});

    ge::Shape conv2d_input_shape({1, 56, 56, 64});
    ge::TensorDesc conv2d_input_tensor_desc(conv2d_input_shape, FORMAT_NHWC, DT_FLOAT);
    conv_op.update_input_desc_x(conv2d_input_tensor_desc);
    conv_op.update_input_desc_filter(filter_desc);
    ge::Shape conv2d_output_shape({1, 28, 28, 64});
    ge::TensorDesc conv2d_output_tensor_desc(conv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
    conv_op.update_output_desc_y(conv2d_output_tensor_desc);

    ge::Operator::OpListInt axis = {0};
    auto squeeze_op = op::Squeeze("squeeze_op");
    squeeze_op.set_input_x(conv_op).set_attr_axis(axis);
    ge::Shape squeeze_input_shape({1, 28, 28, 64});
    ge::TensorDesc squeeze_input_tensor_desc(squeeze_input_shape, FORMAT_NHWC, DT_FLOAT);
    squeeze_op.update_input_desc_x(squeeze_input_tensor_desc);
    ge::Shape squeeze_output_shape({28, 28, 64});
    ge::TensorDesc squeeze_output_tensor_desc(squeeze_output_shape, FORMAT_NHWC, DT_FLOAT);
    squeeze_op.update_output_desc_y(squeeze_output_tensor_desc);


    auto bias_shape = vector<int64_t>({6});
    ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_bias = op::Data("data_bias");
    data_bias.update_input_desc_x(bias_desc);
    data_bias.update_output_desc_y(bias_desc);

    auto biasadd_op = op::BiasAdd("biasadd_op");
    biasadd_op.set_input_x(squeeze_op).set_input_bias(data_bias).set_attr_data_format("NHWC");
    ge::Shape biasadd_input_shape({28, 28, 64});
    ge::TensorDesc biasadd_input_tensor_desc(biasadd_input_shape, FORMAT_NHWC, DT_FLOAT);
    biasadd_op.update_input_desc_x(biasadd_input_tensor_desc);
    ge::Shape biasadd_output_shape({28, 28, 64});
    ge::TensorDesc biasadd_output_tensor_desc(biasadd_output_shape, FORMAT_NHWC, DT_FLOAT);
    biasadd_op.update_output_desc_y(biasadd_output_tensor_desc);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(biasadd_op);

    std::vector<Operator> inputs{conv_input_x_data, filter_0, data_bias};
    std::vector<Operator> outputs{end_op1};
    
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DSqueezeBiasaddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    
    bool passMatchnodea = false;
    bool passMatchshapea = false;
    bool passMatchnodeb = false;
    bool passMatchshapeb = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BiasAdd") {
            vector<int64_t> BiasAdd_input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
            if (BiasAdd_input_shape.size() == 4){
                passMatchshapea = true; 
            }
            auto out_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
            if (out_node->GetType() == "Conv2D"){
               passMatchnodea = true; 
            }
        }
        if (node->GetType() == "Squeeze") {
            vector<int64_t> Squeeze_input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
            if (Squeeze_input_shape.size() == 4){
                passMatchshapeb = true; 
            }
            auto out_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
            if (out_node->GetType() == "BiasAdd"){
                passMatchnodeb = true;
            }
        }
    }
    EXPECT_EQ(passMatchnodea, true);
    EXPECT_EQ(passMatchshapea, true);
    EXPECT_EQ(passMatchnodeb, true);
    EXPECT_EQ(passMatchshapeb, true);
}
TEST_F(conv2d_squeeze_biasadd_test, conv2d_squeeze_biasadd_test_2) {
    ge::Graph graph("conv2d_squeeze_biasadd_test_2");
    auto conv_input_x_data = op::Data("conv_input_x_data");
    ge::TensorDesc tensorDescX(ge::Shape({1, 1, 3200, 256}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    TensorDesc filter_desc(ge::Shape({1, 1, 256, 256}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    filter_desc.SetOriginFormat(FORMAT_NHWC);
    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    float * filter_value = new float[64 * 64];
    filter.SetTensorDesc(filter_desc);
    filter.SetData((uint8_t*)filter_value, 64 * 64 * 4);
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filter_desc);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
           .set_input_filter(filter_0)
           .set_attr_strides({1, 2, 2, 1})
           .set_attr_pads({1, 1, 1, 1});

    ge::Shape conv2d_input_shape({1, 1, 3200, 256});
    ge::TensorDesc conv2d_input_tensor_desc(conv2d_input_shape, FORMAT_NHWC, DT_FLOAT);
    conv_op.update_input_desc_x(conv2d_input_tensor_desc);
    conv_op.update_input_desc_filter(filter_desc);
    ge::Shape conv2d_output_shape({1, 1, 3200, 256});
    ge::TensorDesc conv2d_output_tensor_desc(conv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
    conv_op.update_output_desc_y(conv2d_output_tensor_desc);

    ge::Operator::OpListInt axis = {0};
    auto squeeze_op = op::Squeeze("squeeze_op");
    squeeze_op.set_input_x(conv_op).set_attr_axis(axis);
    ge::Shape squeeze_input_shape({1, 1, 3200, 256});
    ge::TensorDesc squeeze_input_tensor_desc(squeeze_input_shape, FORMAT_NHWC, DT_FLOAT);
    squeeze_op.update_input_desc_x(squeeze_input_tensor_desc);
    ge::Shape squeeze_output_shape({1, 3200, 256});
    ge::TensorDesc squeeze_output_tensor_desc(squeeze_output_shape, FORMAT_NHWC, DT_FLOAT);
    squeeze_op.update_output_desc_y(squeeze_output_tensor_desc);


    auto bias_shape = vector<int64_t>({256});
    ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_bias = op::Data("data_bias");
    data_bias.update_input_desc_x(bias_desc);
    data_bias.update_output_desc_y(bias_desc);

    auto biasadd_op = op::BiasAdd("biasadd_op");
    biasadd_op.set_input_x(squeeze_op).set_input_bias(data_bias).set_attr_data_format("NHWC");
    ge::Shape biasadd_input_shape({1, 3200, 256});
    ge::TensorDesc biasadd_input_tensor_desc(biasadd_input_shape, FORMAT_NHWC, DT_FLOAT);
    biasadd_op.update_input_desc_x(biasadd_input_tensor_desc);
    ge::Shape biasadd_output_shape({1, 3200, 256});
    ge::TensorDesc biasadd_output_tensor_desc(biasadd_output_shape, FORMAT_NHWC, DT_FLOAT);
    biasadd_op.update_output_desc_y(biasadd_output_tensor_desc);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(biasadd_op);

    std::vector<Operator> inputs{conv_input_x_data, filter_0, data_bias};
    std::vector<Operator> outputs{end_op1};
    
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DSqueezeBiasaddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    
    bool passMatchnodea = false;
    bool passMatchshapea = false;
    bool passMatchnodeb = false;
    bool passMatchshapeb = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BiasAdd") {
            vector<int64_t> BiasAdd_input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
            if (BiasAdd_input_shape.size() == 4){
                passMatchshapea = true; 
            }
            auto out_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
            if (out_node->GetType() == "Conv2D"){
               passMatchnodea = true; 
            }
        }
        if (node->GetType() == "Squeeze") {
            vector<int64_t> Squeeze_input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
            if (Squeeze_input_shape.size() == 4){
                passMatchshapeb = true; 
            }
            auto out_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
            if (out_node->GetType() == "BiasAdd"){
                passMatchnodeb = true;
            }
        }
    }
    EXPECT_EQ(passMatchnodea, true);
    EXPECT_EQ(passMatchshapea, true);
    EXPECT_EQ(passMatchnodeb, true);
    EXPECT_EQ(passMatchshapeb, true);
}
TEST_F(conv2d_squeeze_biasadd_test, conv2d_squeeze_biasadd_test_3) {
    ge::Graph graph("conv2d_squeeze_biasadd_test_3");
    auto conv_input_x_data = op::Data("conv_input_x_data");
    ge::TensorDesc tensorDescX(ge::Shape({1, 56, 56, 64}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    TensorDesc filter_desc(ge::Shape({64, 1, 1, 64}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    filter_desc.SetOriginFormat(FORMAT_NHWC);
    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    float * filter_value = new float[64 * 64];
    filter.SetTensorDesc(filter_desc);
    filter.SetData((uint8_t*)filter_value, 64 * 64 * 4);
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filter_desc);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
           .set_input_filter(filter_0)
           .set_attr_strides({1, 2, 2, 1})
           .set_attr_pads({1, 1, 1, 1});

    ge::Shape conv2d_input_shape({1, 56, 56, 64});
    ge::TensorDesc conv2d_input_tensor_desc(conv2d_input_shape, FORMAT_NHWC, DT_FLOAT);
    conv_op.update_input_desc_x(conv2d_input_tensor_desc);
    conv_op.update_input_desc_filter(filter_desc);
    ge::Shape conv2d_output_shape({1, 28, 28, 64});
    ge::TensorDesc conv2d_output_tensor_desc(conv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
    conv_op.update_output_desc_y(conv2d_output_tensor_desc);

    ge::Operator::OpListInt axis = {0};
    auto squeeze_op = op::Squeeze("squeeze_op");
    squeeze_op.set_input_x(conv_op).set_attr_axis(axis);
    ge::Shape squeeze_input_shape({1, 28, 28, 64});
    ge::TensorDesc squeeze_input_tensor_desc(squeeze_input_shape, FORMAT_NHWC, DT_FLOAT);
    squeeze_op.update_input_desc_x(squeeze_input_tensor_desc);
    ge::Shape squeeze_output_shape({28, 28, 64});
    ge::TensorDesc squeeze_output_tensor_desc(squeeze_output_shape, FORMAT_NHWC, DT_FLOAT);
    squeeze_op.update_output_desc_y(squeeze_output_tensor_desc);


    auto bias_shape = vector<int64_t>({1, 28});
    ge::TensorDesc bias_desc(ge::Shape(bias_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_bias = op::Data("data_bias");
    data_bias.update_input_desc_x(bias_desc);
    data_bias.update_output_desc_y(bias_desc);

    auto biasadd_op = op::BiasAdd("biasadd_op");
    biasadd_op.set_input_x(squeeze_op).set_input_bias(data_bias).set_attr_data_format("NHWC");
    ge::Shape biasadd_input_shape({28, 28, 64});
    ge::TensorDesc biasadd_input_tensor_desc(biasadd_input_shape, FORMAT_NHWC, DT_FLOAT);
    biasadd_op.update_input_desc_x(biasadd_input_tensor_desc);
    ge::Shape biasadd_output_shape({28, 28, 64});
    ge::TensorDesc biasadd_output_tensor_desc(biasadd_output_shape, FORMAT_NHWC, DT_FLOAT);
    biasadd_op.update_output_desc_y(biasadd_output_tensor_desc);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(biasadd_op);

    std::vector<Operator> inputs{conv_input_x_data, filter_0, data_bias};
    std::vector<Operator> outputs{end_op1};
    
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    fe::FusionPassTestUtils::RunGraphFusionPass("Conv2DSqueezeBiasaddFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    
    bool passMatchnodea = false;
    bool passMatchshapea = false;
    bool passMatchnodeb = false;
    bool passMatchshapeb = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BiasAdd") {
            vector<int64_t> BiasAdd_input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
            if (BiasAdd_input_shape.size() == 3){
                passMatchshapea = true; 
            }
            auto out_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
            if (out_node->GetType() == "Squeeze"){
               passMatchnodea = true; 
            }
        }
        if (node->GetType() == "Squeeze") {
            vector<int64_t> Squeeze_input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
            if (Squeeze_input_shape.size() == 4){
                passMatchshapeb = true; 
            }
            auto out_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
            if (out_node->GetType() == "Conv2D"){
                passMatchnodeb = true;
            }
        }
    }
    EXPECT_EQ(passMatchnodea, true);
    EXPECT_EQ(passMatchshapea, true);
    EXPECT_EQ(passMatchnodeb, true);
    EXPECT_EQ(passMatchshapeb, true);
}