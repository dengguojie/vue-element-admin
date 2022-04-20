/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "array_ops.h"
#include "common/lx_fusion_func.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "nonlinear_fuc_ops.h"

#define private public
#define protected public
#include "common/inc/op_log.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_slice_utils.h"
#include "fusion_pass_test_utils.h"
#include "inc/common/op_slice_info.h"
#include "common/util/platform_info.h"
#include "transformation_ops.h"

using namespace fe;
using namespace ge;
using namespace op;

class TbeFullyconnectionElemwiseFusionPassTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TbeFullyconnectionElemwiseFusionPassTest SetUp" << std::endl;

    std::map<string, fe::BufferFusionPassRegistry::CreateFn> createFns =
        fe::BufferFusionPassRegistry::GetInstance().GetCreateFnByType(type_fusion_pass);
    const auto &iter = createFns.find(name_fusion_pass);

    if (iter != createFns.end()) {
      ptr_buffer_fusion_pass_func =
          std::unique_ptr<fe::BufferFusionPassBase>(dynamic_cast<fe::BufferFusionPassBase *>(iter->second()));
      EXPECT_NE(ptr_buffer_fusion_pass_func, nullptr);

      ptr_buffer_fusion_pass_func->SetName(name_fusion_pass);
      patterns = ptr_buffer_fusion_pass_func->DefinePatterns();
    }

    EXPECT_NE(patterns.size(), 0);
  }

  static void TearDownTestCase() { std::cout << "TbeFullyconnectionElemwiseFusionPassTest TearDown" << std::endl; }

 private:
  static BufferFusionMapping ConstructFusionMappingOfDoubleOut(const ComputeGraphPtr compute_graph_ptr);
  static BufferFusionMapping ConstructFusionMappingOfAddRelu6(const ComputeGraphPtr compute_graph_ptr);
  static BufferFusionMapping ConstructFusionMappingOfEltwise(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<fe::BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<fe::BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static fe::BufferFusionPassType type_fusion_pass;
};
vector<fe::BufferFusionPattern *> TbeFullyconnectionElemwiseFusionPassTest::patterns;
std::unique_ptr<fe::BufferFusionPassBase> TbeFullyconnectionElemwiseFusionPassTest::ptr_buffer_fusion_pass_func;
const string TbeFullyconnectionElemwiseFusionPassTest::name_fusion_pass = "TbeFullyconnectionElemwiseDequantFusionPass";
const fe::BufferFusionPassType TbeFullyconnectionElemwiseFusionPassTest::type_fusion_pass =
    fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

fe::BufferFusionMapping TbeFullyconnectionElemwiseFusionPassTest::ConstructFusionMappingOfDoubleOut(
    const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_batchmatmul;
  vector<ge::NodePtr> nodes_elemwise;
  vector<ge::NodePtr> nodes_output;

  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "BatchMatMul" or desc->GetType() == "BatchMatMulV2") {
      nodes_batchmatmul.push_back(ptr_node);
    } else if (desc->GetType() == "Relu") {
      nodes_elemwise.push_back(ptr_node);
    } else if (desc->GetType() == "Add") {
      nodes_output.push_back(ptr_node);
    }
  }

  BufferFusionPattern *pattern;
  EXPECT_TRUE(FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeBatchMatMulElemwiseDoubleOut", &pattern));
  OP_LOGD("test TbeFullyconnectionElemwiseFusionPassTest", "desc size(%zu) in pattern(%s)",
          pattern->GetOpDescs().size(), pattern->GetName().c_str());

  BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "FullyConnection/MatMul/BatchMatmul") {
      mapping[desc] = nodes_batchmatmul;
    } else if (desc->desc_name == "eltwise1") {
      mapping[desc] = nodes_elemwise;
    } else if (desc->desc_name == "output") {
      mapping[desc] = nodes_output;
    }
  }

  return mapping;
}

fe::BufferFusionMapping TbeFullyconnectionElemwiseFusionPassTest::ConstructFusionMappingOfAddRelu6(
    const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_fc;
  vector<ge::NodePtr> nodes_elemwise;
  vector<ge::NodePtr> nodes_elemwise1;
  vector<ge::NodePtr> nodes_input;

  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "FullyConnection") {
      nodes_fc.push_back(ptr_node);
    } else if (desc->GetType() == "Add") {
      nodes_elemwise.push_back(ptr_node);
    } else if (desc->GetType() == "Relu6") {
      nodes_elemwise1.push_back(ptr_node);
    } else if (desc->GetName() == "data_c") {
      nodes_input.push_back(ptr_node);
    }
  }

  BufferFusionPattern *pattern;
  EXPECT_TRUE(FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeFullyconnectionAddRelu6FusionPass", &pattern));
  OP_LOGD("test TbeFullyconnectionElemwiseFusionPassTest", "desc size(%zu) in pattern(%s)",
          pattern->GetOpDescs().size(), pattern->GetName().c_str());

  BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "FullyConnection/MatMul/BatchMatmul") {
      mapping[desc] = nodes_fc;
    } else if (desc->desc_name == "eltwise1") {
      mapping[desc] = nodes_elemwise;
    } else if (desc->desc_name == "eltwise2") {
      mapping[desc] = nodes_elemwise1;
    } else if (desc->desc_name == "InputData") {
      mapping[desc] = nodes_input;
    }
  }

  return mapping;
}

fe::BufferFusionMapping TbeFullyconnectionElemwiseFusionPassTest::ConstructFusionMappingOfEltwise(
    const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_matmul;
  vector<ge::NodePtr> nodes_elemwise;
  vector<ge::NodePtr> nodes_input;

  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "MatMulV2") {
      nodes_matmul.push_back(ptr_node);
    } else if (desc->GetType() == "Add") {
      nodes_elemwise.push_back(ptr_node);
    } else if (desc->GetName() == "data_c") {
      nodes_input.push_back(ptr_node);
    }
  }

  BufferFusionPattern *pattern;
  EXPECT_TRUE(FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeFullyconnectionElemwiseDequantFusionPass", &pattern));
  OP_LOGD("test TbeFullyconnectionElemwiseFusionPassTest", "desc size(%zu) in pattern(%s)",
          pattern->GetOpDescs().size(), pattern->GetName().c_str());

  BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "FullyConnection/MatMul/BatchMatmul") {
      mapping[desc] = nodes_matmul;
    } else if (desc->desc_name == "eltwise1") {
      mapping[desc] = nodes_elemwise;
    } else if (desc->desc_name == "InputData") {
      mapping[desc] = nodes_input;
    } else {
      mapping[desc] = {};
    }
  }

  return mapping;
}

TEST_F(TbeFullyconnectionElemwiseFusionPassTest, tbe_fullyconnection_elemwise_fusion_double_out) {
  ge::Graph graph(this->test_info_->name());

  ge::TensorDesc a_desc(ge::Shape({72, 32, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_a = op::Data("data_a");
  data_a.update_input_desc_x(a_desc);
  data_a.update_output_desc_y(a_desc);

  ge::TensorDesc b_desc(ge::Shape({128, 32, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_b = op::Data("data_b");
  data_b.update_input_desc_x(b_desc);
  data_b.update_output_desc_y(b_desc);

  auto batch_matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                             .set_input_x1(data_a)
                             .set_input_x2(data_b)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);
  batch_matmul_op.update_output_desc_y(
      ge::TensorDesc(ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  auto relu_op = op::Relu("Relu").set_input_x(batch_matmul_op);
  relu_op.update_output_desc_y(ge::TensorDesc(ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  ge::TensorDesc c_desc(ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_c = op::Data("data_c");
  data_c.update_input_desc_x(c_desc);
  data_c.update_output_desc_y(c_desc);

  auto add_op = op::Add("add_op").set_input_x1(batch_matmul_op).set_input_x2(data_c);
  add_op.update_output_desc_y(ge::TensorDesc(ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  std::vector<Operator> inputs{data_a, data_b, data_c};
  std::vector<Operator> outputs{add_op, relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  vector<fe::AxisSplitMap> asm_mm{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),  // m
      CreateAxisSplitMap({CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {1})}),  // n
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),  // batch
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"BatchMatMulV2"}));
  vector<fe::AxisSplitMap> asm_relu{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_relu, {"Relu"}));
  vector<fe::AxisSplitMap> asm_add{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1}), CreateInputSplitInfo(1, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2}), CreateInputSplitInfo(1, {2})},
                         {CreateOutputSplitInfo(0, {2})})};
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_add, {"Add"}));

  auto mapping = ConstructFusionMappingOfDoubleOut(compute_graph_ptr);
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            compute_graph_ptr, mapping);
  EXPECT_EQ(res, fe::SUCCESS);

  auto real_split_info_str = GetFusionOpSliceInfoStrFromGraph(compute_graph_ptr);
  auto expect_split_inf_str = CreateFusionOpSliceInfoStrFromSplitMap(
      {CreateAxisSplitMap({CreateInputSplitInfo(0, {2})},
                          {CreateOutputSplitInfo(0, {2}), CreateOutputSplitInfo(1, {2})}),
       CreateAxisSplitMap({CreateInputSplitInfo(1, {0})},
                          {CreateOutputSplitInfo(0, {1}), CreateOutputSplitInfo(1, {1})}),
       CreateAxisSplitMap({CreateInputSplitInfo(0, {0})},
                          {CreateOutputSplitInfo(0, {0}), CreateOutputSplitInfo(1, {0})})});
  EXPECT_EQ(real_split_info_str, expect_split_inf_str);
}

TEST_F(TbeFullyconnectionElemwiseFusionPassTest, tbe_fullyconnection_elemwise_fusion_add_relu6) {
  ge::Graph graph(this->test_info_->name());

  ge::TensorDesc a_desc(ge::Shape({1, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_a = op::Data("data_a");
  data_a.update_input_desc_x(a_desc);
  data_a.update_output_desc_y(a_desc);

  ge::TensorDesc b_desc(ge::Shape({1, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_b = op::Data("data_b");
  data_b.update_input_desc_x(b_desc);
  data_b.update_output_desc_y(b_desc);

  auto batch_matmul_op = op::FullyConnection("FullyConnection").set_input_x(data_a).set_input_w(data_b);

  ge::TensorDesc c_desc(ge::Shape({1, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_c = op::Data("data_c");
  data_c.update_input_desc_x(c_desc);
  data_c.update_output_desc_y(c_desc);

  auto add_op = op::Add("add_op").set_input_x1(batch_matmul_op).set_input_x2(data_c);

  auto relu_op = op::Relu6("relu6").set_input_x(add_op);

  std::vector<Operator> inputs{data_a, data_b, data_c};
  std::vector<Operator> outputs{relu_op};

  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  vector<fe::AxisSplitMap> asm_mm{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"FullyConnection"}));
  vector<fe::AxisSplitMap> asm_add{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1}), CreateInputSplitInfo(1, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2}), CreateInputSplitInfo(1, {2})},
                         {CreateOutputSplitInfo(0, {2})})};
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_add, {"Add"}));
  vector<fe::AxisSplitMap> asm_relu6{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_relu6, {"Relu6"}));

  auto mapping = ConstructFusionMappingOfAddRelu6(compute_graph_ptr);
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            compute_graph_ptr, mapping);
  EXPECT_EQ(res, fe::SUCCESS);

  auto real_split_info_str = GetFusionOpSliceInfoStrFromGraph(compute_graph_ptr);
  auto expect_split_inf_str = CreateFusionOpSliceInfoStrFromSplitMap(
      {CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
       CreateAxisSplitMap({CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})})});
  EXPECT_EQ(real_split_info_str, expect_split_inf_str);
}

TEST_F(TbeFullyconnectionElemwiseFusionPassTest, tbe_fullyconnection_elemwise_fusion_add) {
  ge::Graph graph(this->test_info_->name());

  ge::TensorDesc a_desc(ge::Shape({16, 64}), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto data_a = op::Data("data_a");
  data_a.update_input_desc_x(a_desc);
  data_a.update_output_desc_y(a_desc);

  ge::TensorDesc b_desc(ge::Shape({64, 32}), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto data_b = op::Data("data_b");
  data_b.update_input_desc_x(b_desc);
  data_b.update_output_desc_y(b_desc);

  ge::TensorDesc a_desc_nz(ge::Shape({4, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto transdata_a = op::TransData("transdata_a").set_input_src(data_a);
  transdata_a.update_input_desc_src(a_desc);
  transdata_a.update_output_desc_dst(a_desc_nz);

  ge::TensorDesc b_desc_nz(ge::Shape({2, 4, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto transdata_b = op::TransData("transdata_b").set_input_src(data_b);
  transdata_a.update_input_desc_src(b_desc);
  transdata_a.update_output_desc_dst(b_desc_nz);

  auto matmul_op = op::MatMulV2("MatMulV2")
      .set_input_x1(transdata_a)
      .set_input_x2(transdata_b)
      .set_attr_transpose_x1(false)
      .set_attr_transpose_x1(false)
      .set_attr_offset_x(0);

  ge::TensorDesc c_desc(ge::Shape({16, 32}), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto data_c = op::Data("data_c");
  data_c.update_input_desc_x(c_desc);
  data_c.update_output_desc_y(c_desc);

  auto add_op = op::Add("add_op").set_input_x1(matmul_op).set_input_x2(data_c);

  std::vector<Operator> inputs{data_a, data_b, data_c};
  std::vector<Operator> outputs{add_op};

  graph.SetInputs(inputs).SetOutputs(outputs);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  vector<string> dtype_list;
  dtype_list.push_back("f32");
  dtype_list.push_back("s32");
  dtype_list.push_back("f16");
  std::map<string, vector<string>> intrinsic_map = {{"Intrinsic_fix_pipe_l0c2out", dtype_list}};
  platform_info.ai_core_intrinsic_dtype_map = intrinsic_map;
  opti_compilation_info.soc_version = "soc_version";
  fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  // excute buffer fusion run
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  auto mapping = ConstructFusionMappingOfEltwise(compute_graph_ptr);
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            compute_graph_ptr, mapping);

  // clear soc info
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  EXPECT_EQ(res, fe::SUCCESS);
}
