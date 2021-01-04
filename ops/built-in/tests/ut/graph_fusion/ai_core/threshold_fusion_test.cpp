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
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class threshold_relu_fusion_test:public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout<< "threshold_relu_fusion SetUp" <<std::endl;
    }

    static void TearDownTestCase() {
        std::cout<< "threshold_relu_fusion TearDown" <<std::endl;
    }
};

TEST_F(threshold_relu_fusion_test,threshold_relu_fusion_test_1) {
    //第一部分：使用IR进行构图，注意要对input和output赋属性描述
    ge::Graph graph("threshold_relu_fusion_test_1");
    auto threshold_relu_input_data1 = op::Data("threshold_relu_input_data1");
    std::vector<int64_t> dims{3,4,5};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto threshold_relu_op = op::ThresholdedRelu("threshold_0")
        .set_input_x(threshold_relu_input_data1)
        .set_attr_alpha(-10);
    threshold_relu_op.update_input_desc_x(tensorDesc);
    threshold_relu_op.update_output_desc_y(tensorDesc);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(threshold_relu_op);
    std::vector<Operator> inputs{threshold_relu_input_data1};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //调用融合规则测试的Utils对图进行infershape
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //调用融合Pass，需要指定融合规则名字
    fe::FusionPassTestUtils::RunGraphFusionPass("ThresholdReluPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool thresholdMatch = false;
    bool mulMatch = false;
    for(auto node:compute_graph_ptr->GetAllNodes()) {
        if(node->GetType() == "Threshold" ) {
            thresholdMatch = true;
        }
        if(node->GetType() == "Mul" ) {
            mulMatch = true;
        }
    }
    EXPECT_EQ(thresholdMatch,true);
    EXPECT_EQ(mulMatch,true);
}