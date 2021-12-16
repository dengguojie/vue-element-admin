#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#define private public
#include "fusion_pass_test_utils.h"
#include "common/util/platform_info.h"


using namespace ge;
using namespace op;

class dynamic_gru_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_gru_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_gru_v2_fusion_test TearDown" << std::endl;
    }
};

TEST_F(dynamic_gru_v2_fusion_test, dynamic_gru_v2_fusion_test_1) {
  ge::Graph graph("dynamic_gru_v2_fusion_test_1");
  std::vector<int64_t> dims{32, 32, 32};
  ge::Shape shape(dims);
  std::vector<int64_t> dims1{32, 32, 32};
  ge::Shape shape1(dims1);
  std::vector<int64_t> dims2{32, 32, 32};
  ge::Shape shape2(dims2);
  std::vector<int64_t> dims3{32, 32, 32};
  ge::Shape shape3(dims);
  std::vector<int64_t> dims4{32, 32, 32};
  ge::Shape shape4(dims1);
  std::vector<int64_t> dims5{32, 32, 32};
  ge::Shape shape5(dims2);
  std::vector<int64_t> dims6{32, 32, 32};
  ge::Shape shape6(dims2);
  ge::TensorDesc tensorDesc0(shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc tensorDesc4(shape4, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc tensorDesc5(shape5, ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::TensorDesc tensorDesc6(shape6, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto xData = op::Data("x");
  xData.update_input_desc_x(tensorDesc0);
  xData.update_output_desc_y(tensorDesc0);
  auto weightInputData = op::Data("weight_input");
  weightInputData.update_input_desc_x(tensorDesc1);
  weightInputData.update_output_desc_y(tensorDesc1);
  auto weightHiddenData = op::Data("weight_hidden");
  weightHiddenData.update_input_desc_x(tensorDesc2);
  weightHiddenData.update_output_desc_y(tensorDesc2);
  auto bias_inputData = op::Data("bias_input");
  bias_inputData.update_input_desc_x(tensorDesc3);
  bias_inputData.update_output_desc_y(tensorDesc3);
  auto bias_hiddenData = op::Data("bias_hidden");
  bias_hiddenData.update_input_desc_x(tensorDesc4);
  bias_hiddenData.update_output_desc_y(tensorDesc4);
  auto seq_lengthData = op::Data("seq_length");
  seq_lengthData.update_input_desc_x(tensorDesc5);
  seq_lengthData.update_output_desc_y(tensorDesc5);
  auto init_hData = op::Data("init_h");
  init_hData.update_input_desc_x(tensorDesc6);
  init_hData.update_output_desc_y(tensorDesc6);

  auto dynamicGruV2Op = op::DynamicGRUV2("DynamicGRUV2_01");
  dynamicGruV2Op.set_input_x(xData)
      .set_input_weight_input(weightInputData)
      .set_input_weight_hidden(weightHiddenData);

  std::vector<Operator> inputs{xData, weightInputData, weightHiddenData};
  std::vector<Operator> outputs{dynamicGruV2Op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  opti_compilation_info.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);
  fe::FusionPassTestUtils::RunGraphFusionPass("DynamicGRUV2FusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  bool findDynamicGRUV2 = false;
  for (auto node : computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "DynamicGRUV2") {
      findDynamicGRUV2 = true;
    }
  }
  EXPECT_EQ(findDynamicGRUV2, true);
}

