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

using namespace ge;
using namespace op;

class maxn_fusion_test:public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout<< "inplace_maxn SetUp" <<std::endl;
    }

    static void TearDownTestCase() {
        std::cout<< "inplace_maxn TearDown" <<std::endl;
    }
};

TEST_F(maxn_fusion_test,maxn_fusion_test_1) {
    //第一部分：使用IR进行构图，注意要对input和output赋属性描述
    ge::Graph graph("maxn_fusion_test_1");
    auto maxn_input_data1 = op::Data("maxn_input_data1");
    auto maxn_input_data2 = op::Data("maxn_input_data2");
    auto maxn_input_data3 = op::Data("maxn_input_data3");
    std::vector<int64_t> dims{3};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
    maxn_input_data1.update_input_desc_x(tensorDesc);
    maxn_input_data1.update_output_desc_y(tensorDesc);
    maxn_input_data2.update_input_desc_x(tensorDesc);
    maxn_input_data2.update_output_desc_y(tensorDesc);
    maxn_input_data3.update_input_desc_x(tensorDesc);
    maxn_input_data3.update_output_desc_y(tensorDesc);
    auto maxn_op = op::MaxN("maxn_0")
        .create_dynamic_input_x(3,0)
        .set_dynamic_input_x(0,maxn_input_data1)
        .set_dynamic_input_x(1,maxn_input_data2)
        .set_dynamic_input_x(2,maxn_input_data3);
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(maxn_op);
    std::vector<Operator> inputs{maxn_input_data1, maxn_input_data2, maxn_input_data3};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //调用融合规则测试的Utils对图进行infershape
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //调用融合Pass，需要指定融合规则名字
    fe::FusionPassTestUtils::RunGraphFusionPass("MaxToMaximumPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool maxMatch = false;
    for(auto node:compute_graph_ptr->GetAllNodes()) {
        if(node->GetType() == "Maximum" ) {
            maxMatch = true;
        } 
    }
    EXPECT_EQ(maxMatch,true);
}