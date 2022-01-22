#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "reduce_ops.h"
#include "nonlinear_fuc_ops.h"
#include "transformation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class matmul_reduce_sum_ub_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "matmul_reduce_sum_ub_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matmul_reduce_sum_ub_fusion_test TearDown" << std::endl;
  }
};

namespace fe {
Status RunBufferFusionPass(string fusionPassName, BufferFusionPassType passType,
                           ge::ComputeGraphPtr &compute_graph_ptr) {
  std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
      BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
  const auto &iter = createFns.find(fusionPassName);
  if (iter != createFns.end()) {
    if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
      auto BufferFusionPassBasePtr =
          std::unique_ptr<BufferFusionPassBase>(dynamic_cast<BufferFusionPassBase *>(iter->second()));
      if (BufferFusionPassBasePtr == nullptr) {
        return FAILED;
      }
      ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = compute_graph_ptr->GetAllNodes();
      std::vector<ge::NodePtr> Node_v(NodePtrs.begin(), NodePtrs.end());

      BufferFusionPassBasePtr->SetName(fusionPassName);
      vector<BufferFusionPattern *> patterns = BufferFusionPassBasePtr->DefinePatterns();

      std::vector<BufferFusionOpDesc *> desc = patterns[0]->GetOpDescs();
      vector<ge::NodePtr> ctNodes;
      vector<ge::NodePtr> bmmNodes;
      for (auto i : NodePtrs) {
        auto opDesc = i->GetOpDesc();
        if (opDesc->GetType() == "BatchMatMul" or opDesc->GetType() == "BatchMatMulV2") {
          bmmNodes.push_back(i);
        }
        if (opDesc->GetType() == "ReduceSumD") {
          ctNodes.push_back(i);
        }
      }

      BufferFusionMapping mapping;
      for (auto i : desc) {
        if (i->desc_name == "batch_matmul") {
          mapping[i] = bmmNodes;
        }
        if (i->desc_name == "reduce_sum_d") {
          mapping[i] = ctNodes;
        }
      }
      vector<ge::NodePtr> fusion_nodes;
      auto ret = BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
      return ret;
    }
  }

  return FAILED;
}
}  // namespace fe

TEST_F(matmul_reduce_sum_ub_fusion_test, bmm_reducesumd_test_1) {
  ge::Graph graph("bmm_reducesumd_test_1");

  std::vector<int64_t> dims_x1{2, 32, 15};
  std::vector<int64_t> dims_x2{2, 32, 20};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  Status ret =
      fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(matmul_reduce_sum_ub_fusion_test, bmmV2_reducesumd_test_2) {
  ge::Graph graph("bmmV2_reducesumd_test_2");

  std::vector<int64_t> dims_x1{32, 15};
  std::vector<int64_t> dims_x2{2, 32, 20};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  Status ret =
      fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(matmul_reduce_sum_ub_fusion_test, bmmV2_reducesumd_test_3) {
  ge::Graph graph("bmmV2_reducesumd_test_3");

  std::vector<int64_t> dims_x1{2, 32, 15};
  std::vector<int64_t> dims_x2{32, 20};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  Status ret =
      fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(matmul_reduce_sum_ub_fusion_test, bmm_reducesumd_exception_2output_1) {
  ge::Graph graph("bmm_reducesumd_exception_2output_1");

  std::vector<int64_t> dims_x1{2, 32, 15};
  std::vector<int64_t> dims_x2{2, 32, 20};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  auto relu_op = op::Relu("Relu").set_input_x(matmul_op);
  relu_op.update_input_desc_x(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op, relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
}

TEST_F(matmul_reduce_sum_ub_fusion_test, bmm_reducesumd_exception_batch2_2) {
  ge::Graph graph("bmm_reducesumd_exception_batch2_2");

  std::vector<int64_t> dims_x1{2, 3, 32, 15};
  std::vector<int64_t> dims_x2{2, 3, 32, 20};
  std::vector<int64_t> dims_y{2, 3, 15, 20};
  std::vector<int64_t> dims_reduce_sum{3, 15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
}

TEST_F(matmul_reduce_sum_ub_fusion_test, bmm_reducesumd_exception_output_dtype_3) {
  ge::Graph graph("bmm_reducesumd_exception_output_dtype_3");

  std::vector<int64_t> dims_x1{3, 32, 15};
  std::vector<int64_t> dims_x2{3, 32, 20};
  std::vector<int64_t> dims_y{3, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT16);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT16);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
}

TEST_F(matmul_reduce_sum_ub_fusion_test, bmm_reducesumd_exception_x1_batch1_4) {
  ge::Graph graph("bmm_reducesumd_exception_x1_batch1_4");

  std::vector<int64_t> dims_x1{1, 32, 15};
  std::vector<int64_t> dims_x2{2, 32, 20};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
}

TEST_F(matmul_reduce_sum_ub_fusion_test, bmm_reducesumd_exception_x2_batch1_5) {
  ge::Graph graph("bmm_reducesumd_exception_x2_batch1_5");

  std::vector<int64_t> dims_x1{2, 32, 15};
  std::vector<int64_t> dims_x2{1, 32, 20};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(false);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
}


TEST_F(matmul_reduce_sum_ub_fusion_test, bmm_reducesumd_exception_keep_dims_6) {
  ge::Graph graph("bmm_reducesumd_exception_keep_dims_6");

  std::vector<int64_t> dims_x1{2, 32, 15};
  std::vector<int64_t> dims_x2{2, 32, 20};
  std::vector<int64_t> dims_y{2, 15, 20};
  std::vector<int64_t> dims_reduce_sum{15, 20};
  std::vector<int64_t> axes_value{0};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, ge::FORMAT_ND, ge::DT_FLOAT);
  desc_y.SetOriginShape(shape_y);
  ge::Shape shape_reduce_sum(dims_reduce_sum);
  ge::TensorDesc desc_reduce_sum(shape_reduce_sum, ge::FORMAT_ND, ge::DT_FLOAT);

  auto x1_data = op::Data("x1");
  auto x2_data = op::Data("x2");
  auto matmul_op = op::BatchMatMul("BatchMatMul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_adj_x1(true)
                       .set_attr_adj_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  // create reducesumd
  auto reduce_sum_op = op::ReduceSumD("ReduceSumD")
                           .set_input_x(matmul_op)
                           .set_attr_axes(axes_value)
                           .set_attr_keep_dims(true);
  reduce_sum_op.update_input_desc_x(desc_y);
  reduce_sum_op.update_output_desc_y(desc_reduce_sum);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{reduce_sum_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::RunBufferFusionPass("MatmulReduceSumUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
}