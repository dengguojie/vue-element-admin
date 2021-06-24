#include <string>

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "split_combination_ops.h"
#include "image_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class spatial_transformer_d_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "spatial_transformer_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "spatial_transformer_fusion_test TearDown" << std::endl;
    }
};

TEST_F(spatial_transformer_d_fusion_test, spatial_transformer_d_fusion_test_0) {
    // set soc
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 1;
    platform_info.str_info.ccec_aic_version = "dav-s200";
    opti_compilation_info.soc_version = "SD3403";
    fe::PlatformInfoManager::Instance().platform_info_map_["SD3403"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    ge::Graph graph("spatial_transformer_d_fusion_test_0");

    ge::Tensor inputXDataTensor;
    ge::Tensor inputThetaDataTensor;

    std::vector<int64_t> crops_vec{1, 2, 3, 4};
    ge::Shape shape_x(crops_vec);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
    int64_t input_x_size = tensorDescX.GetShape().GetShapeSize();
    tensorDescX.SetSize(input_x_size * sizeof(float));
    inputXDataTensor.SetTensorDesc(tensorDescX);
    float* input_x_data = nullptr;
    input_x_data = new float[input_x_size];
    *(input_x_data + 0) = 0;
    inputXDataTensor.SetData((uint8_t*)input_x_data, input_x_size * sizeof(float));
    delete[] input_x_data;
    std::string tmp_name0 = "inputXData";
    ge::op::Constant inputXData = op::Constant(tmp_name0).set_attr_value(inputXDataTensor);

    std::vector<int64_t> crops_vec_theta{2};
    ge::Shape shape_theta(crops_vec_theta);
    ge::TensorDesc tensorDescTheta(shape_theta, FORMAT_ND, DT_FLOAT);
    int64_t input_theta_size = tensorDescTheta.GetShape().GetShapeSize();
    tensorDescTheta.SetSize(input_theta_size * sizeof(float));
    inputThetaDataTensor.SetTensorDesc(tensorDescTheta);
    float* input_theta_data = nullptr;
    input_theta_data = new float[input_theta_size];
    *(input_theta_data + 0) = 0;
    inputThetaDataTensor.SetData((uint8_t*)input_theta_data, input_theta_size * sizeof(float));
    delete[] input_theta_data;
    std::string tmp_name1 = "inputThetaData_theta";
    ge::op::Constant inputThetaData = op::Constant(tmp_name1).set_attr_value(inputThetaDataTensor);

    auto stn_layer = op::SpatialTransformerD("stn");
    stn_layer.set_input_x(inputXData);
    stn_layer.set_input_theta(inputThetaData);
    stn_layer.set_attr_output_size({3, 4});
    stn_layer.set_attr_default_theta({1.0, 1.5});
    stn_layer.set_attr_use_default_theta({1, 1, 0, 0, 0, 0});
    stn_layer.set_attr_align_corners(false);
    stn_layer.SetAttr("stn_ori_channel", 2);

    auto end_op = op::Square("end_op");
    end_op.set_input_x(stn_layer);
	
    std::vector<Operator> inputs{};
    inputs.push_back(inputXData);
    inputs.push_back(inputThetaData);
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SpatialTransformerDPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    int total_aicpu_stn_node_num = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SpatialTransformer") {
            total_aicpu_stn_node_num += 1;
        }
    }
    EXPECT_EQ(total_aicpu_stn_node_num, 1);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}
