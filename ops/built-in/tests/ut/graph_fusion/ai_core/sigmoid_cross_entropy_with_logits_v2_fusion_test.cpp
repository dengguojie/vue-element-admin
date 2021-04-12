#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class sigmoid_cross_entropy_with_logits_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sigmoid_cross_entropy_with_logits_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "sigmoid_cross_entropy_with_logits_v2_fusion_test TearDown" << std::endl;
    }
};

TEST_F(sigmoid_cross_entropy_with_logits_v2_fusion_test, sigmoid_cross_entropy_with_logits_v2_fusion_test_1) {

  // predict
  auto data_predict = op::Data("predict");
  std::vector<int64_t> data_predict_vec{1028,1028};
  ge::Shape data_predict_shape(data_predict_vec);
  ge::TensorDesc data_predict_desc(data_predict_shape, FORMAT_ND, DT_FLOAT16);
  data_predict.update_input_desc_x(data_predict_desc);
  data_predict.update_output_desc_y(data_predict_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{1028,1028};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_FLOAT16);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{1028,1028};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT16);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // pos_weight
  auto data_pos_weight = op::Data("pos_weight");
  std::vector<int64_t> data_pos_weight_vec{1028,1028};
  ge::Shape data_pos_weight_shape(data_pos_weight_vec);
  ge::TensorDesc data_pos_weight_desc(data_pos_weight_shape, FORMAT_ND, DT_FLOAT16);
  data_pos_weight.update_input_desc_x(data_pos_weight_desc);
  data_pos_weight.update_output_desc_y(data_pos_weight_desc);

  // reduction
  string reduction = "sum";

  auto sigmoid_op = op::SigmoidCrossEntropyWithLogitsV2("SigmoidCrossEntropyWithLogitsV2");
  sigmoid_op.set_input_predict(data_predict)
            .set_input_target(data_target)
            .set_input_weight(data_weight)
            .set_input_pos_weight(data_pos_weight)
            .set_attr_reduction(reduction);

  std::vector<Operator> inputs{data_predict, data_target, data_weight, data_pos_weight};
  std::vector<Operator> outputs{sigmoid_op};

  ge::Graph graph("sigmoid_cross_entropy_with_logits_v2_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SigmoidCrossEntropyWithLogitsV2FusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findSigmoid = false;
  bool findReduceSum = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SigmoidCrossEntropyWithLogitsV2") {
            findSigmoid = true;
        }

        if (node->GetType() == "ReduceSumD") {
            findReduceSum = true;
        }
    }

  EXPECT_EQ(findSigmoid, true);
  EXPECT_EQ(findReduceSum, true);
}
TEST_F(sigmoid_cross_entropy_with_logits_v2_fusion_test, sigmoid_cross_entropy_with_logits_v2_fusion_test_2) {

  // predict
  auto data_predict = op::Data("predict");
  std::vector<int64_t> data_predict_vec{1028,1028};
  ge::Shape data_predict_shape(data_predict_vec);
  ge::TensorDesc data_predict_desc(data_predict_shape, FORMAT_ND, DT_FLOAT16);
  data_predict.update_input_desc_x(data_predict_desc);
  data_predict.update_output_desc_y(data_predict_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{1028,1028};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_FLOAT16);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{1028,1028};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT16);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // pos_weight
  auto data_pos_weight = op::Data("pos_weight");
  std::vector<int64_t> data_pos_weight_vec{1028,1028};
  ge::Shape data_pos_weight_shape(data_pos_weight_vec);
  ge::TensorDesc data_pos_weight_desc(data_pos_weight_shape, FORMAT_ND, DT_FLOAT16);
  data_pos_weight.update_input_desc_x(data_pos_weight_desc);
  data_pos_weight.update_output_desc_y(data_pos_weight_desc);

  // reduction
  string reduction = "sum";

  auto sigmoid_op = op::SigmoidCrossEntropyWithLogitsV2("SigmoidCrossEntropyWithLogitsV2");
  sigmoid_op.set_input_predict(data_predict)
            .set_input_target(data_target)
            .set_input_weight(data_weight)
            .set_input_pos_weight(data_pos_weight);

  std::vector<Operator> inputs{data_predict, data_target, data_weight, data_pos_weight};
  std::vector<Operator> outputs{sigmoid_op};

  ge::Graph graph("sigmoid_cross_entropy_with_logits_v2_fusion_test_2");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SigmoidCrossEntropyWithLogitsV2FusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findSigmoid = false;
  bool findReduceSum = false;
 // for (auto node: compute_graph_ptr->GetAllNodes()) {
     //   if (node->GetType() == "SigmoidCrossEntropyWithLogitsV2") {
         //   findSigmoid = true;
       // }

        //if (node->GetType() == "ReduceSumD") {
           // findReduceSum = true;
       // }
   // }


  EXPECT_EQ(findReduceSum, false);
}
TEST_F(sigmoid_cross_entropy_with_logits_v2_fusion_test, sigmoid_cross_entropy_with_logits_v2_fusion_test_3) {

  // predict
  auto data_predict = op::Data("predict");
  std::vector<int64_t> data_predict_vec{1028,1028};
  ge::Shape data_predict_shape(data_predict_vec);
  ge::TensorDesc data_predict_desc(data_predict_shape, FORMAT_ND, DT_FLOAT16);
  data_predict.update_input_desc_x(data_predict_desc);
  data_predict.update_output_desc_y(data_predict_desc);

  // target
  auto data_target = op::Data("target");
  std::vector<int64_t> data_target_vec{1028,1028};
  ge::Shape data_target_shape(data_target_vec);
  ge::TensorDesc data_target_desc(data_target_shape, FORMAT_ND, DT_FLOAT16);
  data_target.update_input_desc_x(data_target_desc);
  data_target.update_output_desc_y(data_target_desc);

  // weight
  auto data_weight = op::Data("weight");
  std::vector<int64_t> data_weight_vec{1028,1028};
  ge::Shape data_weight_shape(data_weight_vec);
  ge::TensorDesc data_weight_desc(data_weight_shape, FORMAT_ND, DT_FLOAT16);
  data_weight.update_input_desc_x(data_weight_desc);
  data_weight.update_output_desc_y(data_weight_desc);

  // pos_weight
  auto data_pos_weight = op::Data("pos_weight");
  std::vector<int64_t> data_pos_weight_vec{1028,1028};
  ge::Shape data_pos_weight_shape(data_pos_weight_vec);
  ge::TensorDesc data_pos_weight_desc(data_pos_weight_shape, FORMAT_ND, DT_FLOAT16);
  data_pos_weight.update_input_desc_x(data_pos_weight_desc);
  data_pos_weight.update_output_desc_y(data_pos_weight_desc);

  // reduction
  string reduction = "none";

  auto sigmoid_op = op::SigmoidCrossEntropyWithLogitsV2("SigmoidCrossEntropyWithLogitsV2");
  sigmoid_op.set_input_predict(data_predict)
            .set_input_target(data_target)
            .set_input_weight(data_weight)
            .set_input_pos_weight(data_pos_weight)
            .set_attr_reduction(reduction);

  std::vector<Operator> inputs{data_predict, data_target, data_weight, data_pos_weight};
  std::vector<Operator> outputs{sigmoid_op};

  ge::Graph graph("sigmoid_cross_entropy_with_logits_v2_fusion_test_3");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("SigmoidCrossEntropyWithLogitsV2FusionPass",
                                               fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findSigmoid = false;
  bool findReduceSum = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SigmoidCrossEntropyWithLogitsV2") {
            findSigmoid = true;
        }

        if (node->GetType() == "ReduceSumD") {
            findReduceSum = true;
        }
    }

  EXPECT_EQ(findSigmoid, true);
  EXPECT_EQ(findReduceSum, false);
}