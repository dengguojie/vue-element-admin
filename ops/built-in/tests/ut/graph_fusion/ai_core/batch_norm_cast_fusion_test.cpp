#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "elewise_calculation_ops.h"
#include "state_ops.h"
#include "nn_batch_norm_ops.h"
#include "array_ops.h"
#include "control_flow_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#define protected public
#define private public
#include "graph_fusion/ai_core/fusedbatchnorm_fusion_pass.h"
#undef protected
#undef private

using namespace ge;
using namespace op;

class batch_norm_cast_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "batch_norm_cast_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "batch_norm_cast_fusion_test TearDown" << std::endl;
    }
};

TEST_F(batch_norm_cast_fusion_test, batch_norm_cast_fusion_test_1) {
    ge::Graph graph("batch_norm_cast_fusion_test_1");

    auto bn_input_x_data = op::Data("bn_input_x_data");
    std::vector<int64_t> dims_x{1, 2, 3, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    bn_input_x_data.update_input_desc_x(tensorDescX);
    bn_input_x_data.update_output_desc_y(tensorDescX);

    auto bn_input_scale_data = op::Data("bn_input_scale_data");
    auto bn_input_offset_data = op::Data("bn_input_offset_data");
    std::vector<int64_t> dims_scale{4};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);
    bn_input_scale_data.update_input_desc_x(tensorDescScale);
    bn_input_scale_data.update_output_desc_y(tensorDescScale);

    bn_input_offset_data.update_input_desc_x(tensorDescScale);
    bn_input_offset_data.update_output_desc_y(tensorDescScale);

    auto const_mul = op::Constant("const_mul");
    Tensor consttensor;
    float * dataValue = new float[1];
    * dataValue = 0.1;
    consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC));
    consttensor.SetData((uint8_t*)dataValue, 4);
    const_mul.set_attr_value(consttensor);
    delete []dataValue;

    auto bn_op = op::BatchNorm("batchnorm_0");
    bn_op.set_input_x(bn_input_x_data)
         .set_input_scale(bn_input_scale_data)
         .set_input_offset(bn_input_offset_data)
         .set_attr_is_training(true);

    auto var_mean = op::Variable("var_mean");
    var_mean.update_output_desc_y(tensorDescScale);

    auto cast2_op = op::Cast("cast2_op");
    cast2_op.set_input_x(var_mean)
            .set_attr_dst_type(0);

    auto sub1_op = op::Sub("sub1_op");
    sub1_op.set_input_x1(cast2_op)
           .set_input_x2(bn_op, "batch_mean");

    auto mul1_op = op::Mul("mul1_op");
    mul1_op.set_input_x1(const_mul)
           .set_input_x2(sub1_op);

    auto cast1_op = op::Cast("cast1_op");
    cast1_op.set_input_x(mul1_op)
            .set_attr_dst_type(1);

    auto assignsub1_op = op::AssignSub("assignsub1_op");
    assignsub1_op.set_input_var(var_mean)
                 .set_input_value(cast1_op);
    // ----------------
    auto var_var = op::Variable("var_var");
    var_var.update_output_desc_y(tensorDescScale);

    auto cast4_op = op::Cast("cast4_op");
    cast4_op.set_input_x(var_var)
            .set_attr_dst_type(0);

    auto sub2_op = op::Sub("sub2_op");
    sub2_op.set_input_x1(cast4_op)
           .set_input_x2(bn_op, "batch_variance");

    auto mul2_op = op::Mul("mul2_op");
    mul2_op.set_input_x1(const_mul)
           .set_input_x2(sub2_op);

    auto cast3_op = op::Cast("cast3_op");
    cast3_op.set_input_x(mul2_op)
            .set_attr_dst_type(1);

    auto assignsub2_op = op::AssignSub("assignsub2_op");
    assignsub2_op.set_input_var(var_var)
                 .set_input_value(cast3_op);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(bn_op, "y");

    std::vector<Operator> inputs{bn_input_x_data, bn_input_scale_data, bn_input_offset_data, const_mul, var_mean, var_var};
    std::vector<Operator> outputs{relu_op, assignsub1_op, assignsub2_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("FusedBatchnormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findBnreduce = false;
    bool findBnupdate = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BNTrainingReduce") {
            findBnreduce = true;
        }
        if (node->GetType() == "BNTrainingUpdate") {
            findBnupdate = true;
        }
    }
    EXPECT_EQ(findBnreduce, true);
    EXPECT_EQ(findBnupdate, true);

  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["FusedBatchnormFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["FusedBatchnormFusionPass"].GetEffectTimes(), 1);
}
TEST_F(batch_norm_cast_fusion_test, batch_norm_cast_fusion_test_2) {
    ge::Graph graph("batch_norm_cast_fusion_test_2");

    auto bn_input_x_data = op::Data("bn_input_x_data");
    std::vector<int64_t> dims_x{1, 2, 3, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    bn_input_x_data.update_input_desc_x(tensorDescX);
    bn_input_x_data.update_output_desc_y(tensorDescX);

    auto bn_input_scale_data = op::Data("bn_input_scale_data");
    auto bn_input_offset_data = op::Data("bn_input_offset_data");
    std::vector<int64_t> dims_scale{4};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);
    bn_input_scale_data.update_input_desc_x(tensorDescScale);
    bn_input_scale_data.update_output_desc_y(tensorDescScale);

    bn_input_offset_data.update_input_desc_x(tensorDescScale);
    bn_input_offset_data.update_output_desc_y(tensorDescScale);

    auto const_mul = op::Variable("const_mul");
    const_mul.update_output_desc_y(tensorDescScale);

    auto bn_op = op::BatchNorm("batchnorm_0");
    bn_op.set_input_x(bn_input_x_data)
         .set_input_scale(bn_input_scale_data)
         .set_input_offset(bn_input_offset_data)
         .set_attr_is_training(true);

    auto var_mean = op::Variable("var_mean");
    var_mean.update_output_desc_y(tensorDescScale);

    auto cast2_op = op::Cast("cast2_op");
    cast2_op.set_input_x(var_mean)
            .set_attr_dst_type(0);

    auto sub1_op = op::Sub("sub1_op");
    sub1_op.set_input_x1(cast2_op)
           .set_input_x2(bn_op, "batch_mean");

    auto mul1_op = op::Mul("mul1_op");
    mul1_op.set_input_x1(const_mul)
           .set_input_x2(sub1_op);

    auto cast1_op = op::Cast("cast1_op");
    cast1_op.set_input_x(mul1_op)
            .set_attr_dst_type(1);

    auto assignsub1_op = op::AssignSub("assignsub1_op");
    assignsub1_op.set_input_var(var_mean)
                 .set_input_value(cast1_op);
    // ----------------
    auto var_var = op::Variable("var_var");
    var_var.update_output_desc_y(tensorDescScale);

    auto cast4_op = op::Cast("cast4_op");
    cast4_op.set_input_x(var_var)
            .set_attr_dst_type(0);

    auto sub2_op = op::Sub("sub2_op");
    sub2_op.set_input_x1(cast4_op)
           .set_input_x2(bn_op, "batch_variance");

    auto mul2_op = op::Mul("mul2_op");
    mul2_op.set_input_x1(const_mul)
           .set_input_x2(sub2_op);

    auto cast3_op = op::Cast("cast3_op");
    cast3_op.set_input_x(mul2_op)
            .set_attr_dst_type(1);

    auto assignsub2_op = op::AssignSub("assignsub2_op");
    assignsub2_op.set_input_var(var_var)
                 .set_input_value(cast3_op);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(bn_op, "y");

    std::vector<Operator> inputs{bn_input_x_data, bn_input_scale_data, bn_input_offset_data, const_mul, var_mean, var_var};
    std::vector<Operator> outputs{relu_op, assignsub1_op, assignsub2_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("FusedBatchnormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findBnreduce = false;
    bool findBnupdate = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BNTrainingReduce") {
            findBnreduce = true;
        }
        if (node->GetType() == "BNTrainingUpdate") {
            findBnupdate = true;
        }
    }
    EXPECT_EQ(findBnreduce, false);
    EXPECT_EQ(findBnupdate, false);
}
TEST_F(batch_norm_cast_fusion_test, batch_norm_cast_fusion_test_3) {
  ge::Graph graph("batch_norm_cast_fusion_test_3");

  auto bn_input_x_data = op::Data("bn_input_x_data");
  std::vector<int64_t> dims_x{1, 2, 3, 4};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
  bn_input_x_data.update_input_desc_x(tensorDescX);
  bn_input_x_data.update_output_desc_y(tensorDescX);

  auto bn_input_scale_data = op::Data("bn_input_scale_data");
  auto bn_input_offset_data = op::Data("bn_input_offset_data");
  std::vector<int64_t> dims_scale{4};
  ge::Shape shape_scale(dims_scale);
  ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC, DT_FLOAT);
  bn_input_scale_data.update_input_desc_x(tensorDescScale);
  bn_input_scale_data.update_output_desc_y(tensorDescScale);

  bn_input_offset_data.update_input_desc_x(tensorDescScale);
  bn_input_offset_data.update_output_desc_y(tensorDescScale);

  auto const_mul = op::Constant("const_mul");
  Tensor consttensor;
  float* dataValue = new float[1];
  *dataValue = 0.1;
  consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_FLOAT16));
  consttensor.SetData((uint8_t*)dataValue, 4);
  const_mul.set_attr_value(consttensor);
  delete[] dataValue;

  auto bn_op = op::BatchNorm("batchnorm_0");
  bn_op.set_input_x(bn_input_x_data)
      .set_input_scale(bn_input_scale_data)
      .set_input_offset(bn_input_offset_data)
      .set_attr_is_training(true);

  auto var_mean = op::Variable("var_mean");
  var_mean.update_output_desc_y(tensorDescScale);

  auto cast2_op = op::Cast("cast2_op");
  cast2_op.set_input_x(var_mean).set_attr_dst_type(0);

  auto sub1_op = op::Sub("sub1_op");
  sub1_op.set_input_x1(cast2_op).set_input_x2(bn_op, "batch_mean");

  auto mul1_op = op::Mul("mul1_op");
  mul1_op.set_input_x1(const_mul).set_input_x2(sub1_op);

  auto cast1_op = op::Cast("cast1_op");
  cast1_op.set_input_x(mul1_op).set_attr_dst_type(1);

  auto assignsub1_op = op::AssignSub("assignsub1_op");
  assignsub1_op.set_input_var(var_mean).set_input_value(cast1_op);
  // ----------------
  auto var_var = op::Variable("var_var");
  var_var.update_output_desc_y(tensorDescScale);

  auto cast4_op = op::Cast("cast4_op");
  cast4_op.set_input_x(var_var).set_attr_dst_type(0);

  auto sub2_op = op::Sub("sub2_op");
  sub2_op.set_input_x1(cast4_op).set_input_x2(bn_op, "batch_variance");

  auto mul2_op = op::Mul("mul2_op");
  mul2_op.set_input_x1(const_mul).set_input_x2(sub2_op);

  auto cast3_op = op::Cast("cast3_op");
  cast3_op.set_input_x(mul2_op).set_attr_dst_type(1);

  auto assignsub2_op = op::AssignSub("assignsub2_op");
  assignsub2_op.set_input_var(var_var).set_input_value(cast3_op);

  auto relu_op = op::Relu("relu_op");
  relu_op.set_input_x(bn_op, "y");


  std::vector<Operator> inputs{bn_input_x_data, bn_input_scale_data, bn_input_offset_data, const_mul, var_mean,
                               var_var};
  std::vector<Operator> outputs{relu_op, assignsub1_op, assignsub2_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchNorm") {
      fe::FusedBatchnormFusionPass a;
      fe::PassMatchResult matchResult;
      a.FindOutputNodeByName(node, "relu_op", matchResult, true);
    }
  }
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("FusedBatchnormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findBnreduce = false;
  bool findBnupdate = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BNTrainingReduce") {
      findBnreduce = true;
    }
    if (node->GetType() == "BNTrainingUpdate") {
      findBnupdate = true;
    }
  }
  EXPECT_EQ(findBnreduce, true);
  EXPECT_EQ(findBnupdate, true);

  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder& fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["FusedBatchnormFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["FusedBatchnormFusionPass"].GetEffectTimes(), 1);
}

