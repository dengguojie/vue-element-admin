#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nn_pooling_ops.h"

using namespace ge;
using namespace op;

class max_pool3d_grad_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "max_pool3d_grad_grad_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "max_pool3d_grad_grad_fusion_test TearDown" << std::endl;
    }
};

TEST_F(max_pool3d_grad_grad_fusion_test, max_pool3d_grad_grad_fusion_test_1) {
    ge::Graph graph("max_pool3d_grad_grad_fusion_test_1");
    auto xdata1 = op::Data("x1");
    std::vector<int64_t> dims1{2, 2, 17, 13, 16};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    xdata1.update_input_desc_x(tensorDesc1);
    xdata1.update_output_desc_y(tensorDesc1);

    auto xdata2 = op::Data("x2");
    std::vector<int64_t> dims2{2, 2, 17, 13, 16};
    ge::Shape shape2(dims2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    xdata2.update_input_desc_x(tensorDesc2);
    xdata2.update_output_desc_y(tensorDesc2);

    auto xdata3 = op::Data("x3");
    std::vector<int64_t> dims3{2, 2, 17, 13, 16};
    ge::Shape shape3(dims3);
    ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    xdata3.update_input_desc_x(tensorDesc3);
    xdata3.update_output_desc_y(tensorDesc3);

    auto maxpool3dgradgrad_op = op::MaxPool3DGradGrad("maxpool3dgradgrad_op");
    maxpool3dgradgrad_op.set_input_orig_x(xdata1)
                        .set_input_orig_y(xdata2)
                        .set_input_grads(xdata3)
                        .set_attr_data_format("NDHWC")
                        .set_attr_strides({1,1,1,1,1})
                        .set_attr_ksize({1,1,1,1,1})
                        .set_attr_pads({1,1,1,1,1});

    std::vector<Operator> inputs{xdata1,xdata2,xdata3};
    std::vector<Operator> outputs{maxpool3dgradgrad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MaxPool3DGradGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMaxPool3DGradGradD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "MaxPool3DGradGradD") {
            findMaxPool3DGradGradD = true;
        }
    }
    EXPECT_EQ(findMaxPool3DGradGradD, true);
}
TEST_F(max_pool3d_grad_grad_fusion_test, max_pool3d_grad_grad_fusion_test_2) {
    ge::Graph graph("max_pool3d_grad_grad_fusion_test_2");
    auto xdata1 = op::Data("x1");
    std::vector<int64_t> dims1{2, 2, 17, 13, 16};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCDHW, ge::DT_FLOAT);
    xdata1.update_input_desc_x(tensorDesc1);
    xdata1.update_output_desc_y(tensorDesc1);

    auto xdata2 = op::Data("x2");
    std::vector<int64_t> dims2{2, 2, 17, 13, 16};
    ge::Shape shape2(dims2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_NCDHW, ge::DT_FLOAT);
    xdata2.update_input_desc_x(tensorDesc2);
    xdata2.update_output_desc_y(tensorDesc2);

    auto xdata3 = op::Data("x3");
    std::vector<int64_t> dims3{2, 2, 17, 13, 16};
    ge::Shape shape3(dims3);
    ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_NCDHW, ge::DT_FLOAT);
    xdata3.update_input_desc_x(tensorDesc3);
    xdata3.update_output_desc_y(tensorDesc3);

    auto maxpool3dgradgrad_op = op::MaxPool3DGradGrad("maxpool3dgradgrad_op");
    maxpool3dgradgrad_op.set_input_orig_x(xdata1)
                        .set_input_orig_y(xdata2)
                        .set_input_grads(xdata3)
                        .set_attr_data_format("NDHWC")
                        .set_attr_strides({1,1,1,1,1})
                        .set_attr_ksize({1,1,1,1,1})
                        .set_attr_pads({1,1,1,1,1});

    std::vector<Operator> inputs{xdata1,xdata2,xdata3};
    std::vector<Operator> outputs{maxpool3dgradgrad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MaxPool3DGradGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMaxPool3DGradGradD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "MaxPool3DGradGradD") {
            findMaxPool3DGradGradD = true;
        }
    }
    EXPECT_EQ(findMaxPool3DGradGradD, true);
}