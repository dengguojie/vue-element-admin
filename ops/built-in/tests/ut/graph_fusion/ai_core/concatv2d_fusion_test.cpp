#include <string>

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "split_combination_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class concatv2d_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "concatv2d_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "concatv2d_fusion_test TearDown" << std::endl;
    }
};

TEST_F(concatv2d_fusion_test, concatv2d_fusion_test_1) {
    ge::Graph graph("concatv2d_fusion_test_1");

    ge::Tensor inputx0DataTensor[30];
    ge::op::Constant inputx0Data[30];
    for (uint32_t i = 0; i < 30; i++) {
        std::vector<int64_t> crops_vec0{1};
        ge::Shape shape_x0(crops_vec0);
        ge::TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
        int64_t inputx0_size = tensorDescX0.GetShape().GetShapeSize();
        tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
        inputx0DataTensor[i].SetTensorDesc(tensorDescX0);
        int64_t* inputx0_data = nullptr;
        inputx0_data = new int64_t[inputx0_size];
        *(inputx0_data + 0) = 0;
        inputx0DataTensor[i].SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
        delete[] inputx0_data;
        std::string tmp_name = "inputx0Data_" + std::to_string(i);
        inputx0Data[i] = op::Constant(tmp_name).set_attr_value(inputx0DataTensor[i]);
    }

    ge::Tensor inputx1DataTensor[30];
    ge::op::Constant inputx1Data[30];
    for (uint32_t i = 0; i < 30; i++) {
        std::vector<int64_t> crops_vec1{1};
        ge::Shape shape_x1(crops_vec1);
        ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
        int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
        tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
        inputx1DataTensor[i].SetTensorDesc(tensorDescX1);
        int64_t* inputx1_data = nullptr;
        inputx1_data = new int64_t[inputx1_size];
        *(inputx1_data + 0) = 0;
        inputx1DataTensor[i].SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
        delete[] inputx1_data;
        std::string tmp_name = "inputx1Data_" + std::to_string(i);
        inputx1Data[i] = op::Constant(tmp_name).set_attr_value(inputx1DataTensor[i]);
    }

    auto concat_layer = op::ConcatV2D("concatv2d1");
    concat_layer.create_dynamic_input_x(30);
    for (int64_t n = 0; n < 30; n++) {
        concat_layer.set_dynamic_input_x(n, inputx0Data[n]);
    }
    concat_layer.set_attr_concat_dim(0);
    concat_layer.set_attr_N(30);

    auto concat_layer2 = op::ConcatV2D("concatv2d2");
    concat_layer2.create_dynamic_input_x(31);
    concat_layer2.set_dynamic_input_x(0, concat_layer);
    for (int64_t n = 0; n < 30; n++) {
        concat_layer2.set_dynamic_input_x(n + 1, inputx1Data[n]);
    }
    concat_layer2.set_attr_concat_dim(0);
    concat_layer2.set_attr_N(31);

    auto end_op = op::Square("end_op");
    end_op.set_input_x(concat_layer2);
	
    std::vector<Operator> inputs{};
    for (int64_t n = 0; n < 30; n++) {
        inputs.push_back(inputx0Data[n]);
    }
    for (int64_t n = 0; n < 30; n++) {
        inputs.push_back(inputx1Data[n]);
    }
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    int total_cocncat_node_num = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ConcatV2D") {
            total_cocncat_node_num += 1;
        }
    }
    EXPECT_EQ(total_cocncat_node_num, 1);
}

TEST_F(concatv2d_fusion_test, concatv2d_fusion_test_2) {
    ge::Graph graph("concatv2d_fusion_test_2");

    ge::Tensor inputx0DataTensor[30];
    ge::op::Constant inputx0Data[30];
    for (uint32_t i = 0; i < 30; i++) {
        std::vector<int64_t> crops_vec0{1};
        ge::Shape shape_x0(crops_vec0);
        ge::TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
        int64_t inputx0_size = tensorDescX0.GetShape().GetShapeSize();
        tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
        inputx0DataTensor[i].SetTensorDesc(tensorDescX0);
        int64_t* inputx0_data = nullptr;
        inputx0_data = new int64_t[inputx0_size];
        *(inputx0_data + 0) = 0;
        inputx0DataTensor[i].SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
        delete[] inputx0_data;
        std::string tmp_name = "inputx0Data_" + std::to_string(i);
        inputx0Data[i] = op::Constant(tmp_name).set_attr_value(inputx0DataTensor[i]);
    }

    ge::Tensor inputx1DataTensor[40];
    ge::op::Constant inputx1Data[40];
    for (uint32_t i = 0; i < 40; i++) {
        std::vector<int64_t> crops_vec1{1};
        ge::Shape shape_x1(crops_vec1);
        ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
        int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
        tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
        inputx1DataTensor[i].SetTensorDesc(tensorDescX1);
        int64_t* inputx1_data = nullptr;
        inputx1_data = new int64_t[inputx1_size];
        *(inputx1_data + 0) = 0;
        inputx1DataTensor[i].SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
        delete[] inputx1_data;
        std::string tmp_name = "inputx1Data_" + std::to_string(i);
        inputx1Data[i] = op::Constant(tmp_name).set_attr_value(inputx1DataTensor[i]);
    }

    auto concat_layer = op::ConcatV2D("concatv2d1");
    concat_layer.create_dynamic_input_x(30);
    for (int64_t n = 0; n < 30; n++) {
        concat_layer.set_dynamic_input_x(n, inputx0Data[n]);
    }
    concat_layer.set_attr_concat_dim(0);
    concat_layer.set_attr_N(30);

    auto concat_layer2 = op::ConcatV2D("concatv2d2");
    concat_layer2.create_dynamic_input_x(41);
    concat_layer2.set_dynamic_input_x(0, concat_layer);
    for (int64_t n = 0; n < 40; n++) {
        concat_layer2.set_dynamic_input_x(n + 1, inputx1Data[n]);
    }
    concat_layer2.set_attr_concat_dim(0);
    concat_layer2.set_attr_N(41);

    auto end_op = op::Square("end_op");
    end_op.set_input_x(concat_layer2);
	
    std::vector<Operator> inputs{};
    for (int64_t n = 0; n < 30; n++) {
        inputs.push_back(inputx0Data[n]);
    }
    for (int64_t n = 0; n < 40; n++) {
        inputs.push_back(inputx1Data[n]);
    }
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    int total_cocncat_node_num = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ConcatV2D") {
            total_cocncat_node_num += 1;
        }
    }
    EXPECT_EQ(total_cocncat_node_num, 2);
}
