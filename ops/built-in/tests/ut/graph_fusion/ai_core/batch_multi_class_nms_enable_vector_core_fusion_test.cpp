#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_detect_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "selection_ops.h"
#include "nn_norm_ops.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class batch_multi_class_nms_vector_core_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "batch_multi_class_nms_vector_core_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    std::cout << "batch_multi_class_nms_vector_core_fusion_test TearDown" << std::endl;
  }
};

TEST_F(batch_multi_class_nms_vector_core_fusion_test, batch_multi_class_nms_vector_core_fusion_test_1) {
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 8;
  platform_info.soc_info.vector_core_cnt = 7;
  opti_compilation_info.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  ge::Graph graph("batch_multi_class_nms_vector_core_fusion_test_1");

  auto boxesData = op::Data("boxesData");
  std::vector<int64_t> dims_x{15, 1024, 1, 4};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT16);
  boxesData.update_input_desc_x(tensorDescX);
  boxesData.update_output_desc_y(tensorDescX);

  auto scoresData = op::Data("scoresData");
  std::vector<int64_t> dims_score{15, 1024, 4};
  ge::Shape shape_score(dims_score);
  ge::TensorDesc tensorDescScore(shape_score, FORMAT_ND, DT_FLOAT16);
  scoresData.update_input_desc_x(tensorDescScore);
  scoresData.update_output_desc_y(tensorDescScore);

  auto nmsOp = op::BatchMultiClassNonMaxSuppression("BatchMultiClassNonMaxSuppression_1");
  nmsOp.set_input_boxes(boxesData);
  nmsOp.set_input_scores(scoresData);
  nmsOp.set_attr_score_threshold(0.6);
  nmsOp.set_attr_iou_threshold(0.6);
  nmsOp.set_attr_max_size_per_class(100);
  nmsOp.set_attr_max_total_size(100);

  std::vector<Operator> inputs{boxesData, scoresData};
  std::vector<Operator> outputs{nmsOp};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  auto ret = fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMultiClassNonMaxSuppressionFusionPass",
                                                    fe::BUILT_IN_GRAPH_PASS,
                                                    *compute_graph_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = fe::FusionPassTestUtils::RunGraphFusionPass("BatchMultiClassNonMaxSuppressionFusionPass4VectorCore",
                                                    fe::BUILT_IN_GRAPH_PASS,
                                                    *compute_graph_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  int op_count = 0;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchMultiClassNonMaxSuppression") {
      op_count++;
      auto output3_shape = node->GetOpDesc()->GetOutputDesc(3).GetShape().GetDims();
      EXPECT_EQ(output3_shape.size(), 2);
      EXPECT_EQ(output3_shape[1], 8);
    }
  }
  EXPECT_EQ(op_count, 2);
}

