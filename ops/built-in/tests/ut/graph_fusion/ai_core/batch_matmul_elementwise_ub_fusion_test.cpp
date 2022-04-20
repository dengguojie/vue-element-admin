/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

using namespace fe;
using namespace ge;
using namespace op;

class BatchMatMulElementwiseUbFusionTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SetUp BatchMatMulElementwiseUbFusionTest" << std::endl;
    std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
        BufferFusionPassRegistry::GetInstance().GetCreateFnByType(type_fusion_pass);
    const auto &iter = createFns.find(name_fusion_pass);

    if (iter != createFns.end()) {
      ptr_buffer_fusion_pass_func =
          std::unique_ptr<BufferFusionPassBase>(dynamic_cast<BufferFusionPassBase *>(iter->second()));
      EXPECT_NE(ptr_buffer_fusion_pass_func, nullptr);

      ptr_buffer_fusion_pass_func->SetName(name_fusion_pass);
      patterns = ptr_buffer_fusion_pass_func->DefinePatterns();
    }

    EXPECT_NE(patterns.size(), 0);
  }

  static void TearDownTestCase() { std::cout << "TearDown BatchMatMulElementwiseUbFusionTest" << std::endl; }

 private:
  static BufferFusionMapping ConstructFusionMappingOfElemwise(const ComputeGraphPtr compute_graph_ptr);
  static BufferFusionMapping ConstructFusionMappingOfMulSigmoidMul(const ComputeGraphPtr compute_graph_ptr);
  static BufferFusionMapping ConstructFusionMappingOfElemElem(const ComputeGraphPtr compute_graph_ptr);

  static std::unique_ptr<BufferFusionPassBase> ptr_buffer_fusion_pass_func;
  static vector<BufferFusionPattern *> patterns;
  const static string name_fusion_pass;
  const static BufferFusionPassType type_fusion_pass;
};
vector<BufferFusionPattern *> BatchMatMulElementwiseUbFusionTest::patterns;
std::unique_ptr<BufferFusionPassBase> BatchMatMulElementwiseUbFusionTest::ptr_buffer_fusion_pass_func;
const string BatchMatMulElementwiseUbFusionTest::name_fusion_pass = "TbeBatchMatmulElementWiseFusionPass";
const BufferFusionPassType BatchMatMulElementwiseUbFusionTest::type_fusion_pass = BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;

BufferFusionMapping BatchMatMulElementwiseUbFusionTest::ConstructFusionMappingOfElemwise(
    const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_matmul;
  vector<ge::NodePtr> nodes_elemwise;

  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "BatchMatMul" or desc->GetType() == "BatchMatMulV2") {
      nodes_matmul.push_back(ptr_node);
    } else if (desc->GetType() == "FusedMulAdd") {
      nodes_elemwise.push_back(ptr_node);
    }
  }

  BufferFusionPattern *pattern;
  EXPECT_TRUE(FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeBatchMatmulElemPattern", &pattern));
  OP_LOGD("test batch_matmul_elementwise_ub_fusion", "desc size(%zu) in pattern(%s)", pattern->GetOpDescs().size(),
          pattern->GetName().c_str());

  BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    OP_LOGD("test batch_matmul_elementwise_ub_fusion", "desc(%s) in pattern(%s)", desc->desc_name.c_str(),
            pattern->GetName().c_str());
    if (desc->desc_name == "batchmatmul") {
      mapping[desc] = nodes_matmul;
      OP_LOGD("test batch_matmul_elementwise_ub_fusion", "desc(%s) match %zu nodes", desc->desc_name.c_str(),
              mapping[desc].size());
    } else if (desc->desc_name == "elemwise") {
      mapping[desc] = nodes_elemwise;
    }
  }

  return mapping;
}

BufferFusionMapping BatchMatMulElementwiseUbFusionTest::ConstructFusionMappingOfMulSigmoidMul(
    const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_matmul;
  vector<ge::NodePtr> nodes_elemwise;
  vector<ge::NodePtr> nodes_elemwise1;
  vector<ge::NodePtr> nodes_elemwise2;
  vector<ge::NodePtr> nodes_input;
  vector<ge::NodePtr> nodes_input1;
  vector<ge::NodePtr> nodes_output;

  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "BatchMatMul" or desc->GetType() == "BatchMatMulV2") {
      nodes_matmul.push_back(ptr_node);
      nodes_input1.push_back(ptr_node);
    } else if (desc->GetName() == "mul0") {
      nodes_elemwise.push_back(ptr_node);
      nodes_output.push_back(ptr_node);
    } else if (desc->GetName() == "mul1") {
      nodes_elemwise1.push_back(ptr_node);
    } else if (desc->GetType() == "Sigmoid") {
      nodes_elemwise2.push_back(ptr_node);
    } else if (desc->GetName() == "input") {
      nodes_input.push_back(ptr_node);
    }
  }

  BufferFusionPattern *pattern;
  EXPECT_TRUE(FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeBatchMatmulMulSigmoidMulPattern", &pattern));

  BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "batchmatmul") {
      mapping[desc] = nodes_matmul;
    } else if (desc->desc_name == "elemwise") {
      mapping[desc] = nodes_elemwise;
    } else if (desc->desc_name == "elemwise1") {
      mapping[desc] = nodes_elemwise1;
    } else if (desc->desc_name == "elemwise2") {
      mapping[desc] = nodes_elemwise2;
    } else if (desc->desc_name == "InputData") {
      mapping[desc] = nodes_input;
    } else if (desc->desc_name == "InputData1") {
      mapping[desc] = nodes_input1;
    } else if (desc->desc_name == "OutputData") {
      mapping[desc] = nodes_output;
    }
  }

  return mapping;
}

BufferFusionMapping BatchMatMulElementwiseUbFusionTest::ConstructFusionMappingOfElemElem(
    const ComputeGraphPtr compute_graph_ptr) {
  vector<ge::NodePtr> nodes_matmul;
  vector<ge::NodePtr> nodes_elemwise;
  vector<ge::NodePtr> nodes_elemwise1;

  for (const auto &ptr_node : compute_graph_ptr->GetAllNodes()) {
    auto desc = ptr_node->GetOpDesc();
    if (desc->GetType() == "BatchMatMul" or desc->GetType() == "BatchMatMulV2") {
      nodes_matmul.push_back(ptr_node);
    } else if (desc->GetName() == "add_0") {
      nodes_elemwise.push_back(ptr_node);
    } else if (desc->GetName() == "add_1") {
      nodes_elemwise1.push_back(ptr_node);
    }
  }

  BufferFusionPattern *pattern;
  EXPECT_TRUE(FusionPassTestUtils::GetBufferFusionPattern(patterns, "TbeBatchMatmulElemElemPattern", &pattern));

  BufferFusionMapping mapping;
  for (const auto &desc : pattern->GetOpDescs()) {
    if (desc->desc_name == "batchmatmul") {
      mapping[desc] = nodes_matmul;
    } else if (desc->desc_name == "elemwise") {
      mapping[desc] = nodes_elemwise;
    } else if (desc->desc_name == "elemwise1") {
      mapping[desc] = nodes_elemwise1;
    }
  }

  return mapping;
}

op::Data CreateData(const string &name, const ge::Shape &shape, const ge::Format &format, const ge::DataType &dtype) {
  ge::TensorDesc desc(shape, format, dtype);
  auto data = op::Data(name);
  data.update_input_desc_x(desc);
  data.update_output_desc_y(desc);
  return data;
}

TEST_F(BatchMatMulElementwiseUbFusionTest, batch_matmul_fused_mul_add) {
  ge::Graph graph(this->test_info_->name());

  // step1: Construct Graph
  auto data_a = CreateData("data_a", ge::Shape({4, 16, 2, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_b = CreateData("data_b", ge::Shape({4, 16, 1, 2, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_c = CreateData("data_c", ge::Shape({4, 1, 1, 16}), ge::FORMAT_FRACTAL_NZ, DT_FLOAT16);

  auto const_add = op::Const("const_add");
  Tensor const_add_tensor;
  float *const_add_tensor_value = new float[1];
  for (int i = 0; i < 1; i++) {
    *(const_add_tensor_value + i) = 0.1;
  }
  const_add_tensor.SetData((uint8_t *)const_add_tensor_value, 1 * 4);
  std::vector<int64_t> dims_add{1};
  ge::Shape shape_add(dims_add);
  ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT);
  const_add_tensor.SetTensorDesc(tensorDescAdd);
  const_add.set_attr_value(const_add_tensor);

  auto cast_op = op::Cast("cast_op");
  cast_op.set_input_x(const_add).set_attr_dst_type(1);  // dst_type 1: float16

  auto batch_matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                             .set_input_x1(data_a)
                             .set_input_x2(data_b)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);
  batch_matmul_op.update_output_desc_y(
      ge::TensorDesc(ge::Shape({4, 16, 1, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  // not implement infershape
  auto fused_mul_add_op =
      op::FusedMulAdd("FusedMulAdd").set_input_x1(batch_matmul_op).set_input_x2(cast_op).set_input_x3(data_c);
  fused_mul_add_op.update_output_desc_y(
      ge::TensorDesc(ge::Shape({4, 16, 1, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  std::vector<Operator> inputs{data_a, data_b, data_c};
  std::vector<Operator> outputs{fused_mul_add_op};

  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  // step2: Set op slice of single op
  vector<AxisSplitMap> asm_fused_mul_add{CreateAxisSplitMap(
      {CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(2, {0})}, {CreateOutputSplitInfo(0, {0})})};
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_fused_mul_add, {"FusedMulAdd"}));
  vector<AxisSplitMap> asm_mm{CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})},
                                                 {CreateOutputSplitInfo(0, {0})})};
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"BatchMatMulV2"}));

  // step3: Construct BufferFusionMapping
  auto mapping = ConstructFusionMappingOfElemwise(compute_graph_ptr);

  // step4: Run BufferFusion
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            compute_graph_ptr, mapping);
  EXPECT_EQ(res, fe::SUCCESS);

  // step5: Compare op slice info
  auto real_split_info_str = GetFusionOpSliceInfoStrFromGraph(compute_graph_ptr);
  auto expect_split_inf_str = CreateFusionOpSliceInfoStrFromSplitMap(
      {CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0}), CreateInputSplitInfo(3, {0})},
                          {CreateOutputSplitInfo(0, {0})})});
  EXPECT_EQ(real_split_info_str, expect_split_inf_str);
}

TEST_F(BatchMatMulElementwiseUbFusionTest, batch_matmul_fused_mul_add_without_split_info) {
  ge::Graph graph(this->test_info_->name());

  // step1: Construct Graph
  auto data_a = CreateData("data_a", ge::Shape({1, 12, 8, 8, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_b = CreateData("data_b", ge::Shape({1, 12, 8, 8, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_c = CreateData("data_c", ge::Shape({1, 1, 8, 8, 16, 16}), ge::FORMAT_FRACTAL_NZ, DT_FLOAT16);

  auto const_add = op::Const("const_add");
  Tensor const_add_tensor;
  float *const_add_tensor_value = new float[1];
  for (int i = 0; i < 1; i++) {
    *(const_add_tensor_value + i) = 0.1;
  }
  const_add_tensor.SetData((uint8_t *)const_add_tensor_value, 1 * 4);
  std::vector<int64_t> dims_add{1};
  ge::Shape shape_add(dims_add);
  ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT);
  const_add_tensor.SetTensorDesc(tensorDescAdd);
  const_add.set_attr_value(const_add_tensor);

  auto cast_op = op::Cast("cast_op");
  cast_op.set_input_x(const_add).set_attr_dst_type(1);  // dst_type 1: float16

  auto batch_matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                             .set_input_x1(data_a)
                             .set_input_x2(data_b)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);
  batch_matmul_op.update_output_desc_y(
      ge::TensorDesc(ge::Shape({1, 12, 8, 8, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  // not implement infershape
  auto fused_mul_add_op =
      op::FusedMulAdd("FusedMulAdd").set_input_x1(cast_op).set_input_x2(batch_matmul_op).set_input_x3(data_c);
  fused_mul_add_op.update_output_desc_y(
      ge::TensorDesc(ge::Shape({1, 12, 8, 8, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  std::vector<Operator> inputs{data_a, data_b, data_c};
  std::vector<Operator> outputs{fused_mul_add_op};

  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  // step2: Set op slice of single op
  vector<AxisSplitMap> asm_fused_mul_add{};
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_fused_mul_add, {"FusedMulAdd"}));
  vector<AxisSplitMap> asm_mm{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {3})}, {CreateOutputSplitInfo(0, {3})}),
      CreateAxisSplitMap({CreateInputSplitInfo(1, {2})}, {CreateOutputSplitInfo(0, {2})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1}), CreateInputSplitInfo(1, {1})}, {CreateOutputSplitInfo(0, {1})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"BatchMatMulV2"}));

  // step3: Construct BufferFusionMapping
  auto mapping = ConstructFusionMappingOfElemwise(compute_graph_ptr);

  // step4: Run BufferFusion
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            compute_graph_ptr, mapping);
  EXPECT_EQ(res, fe::SUCCESS);

  // step5: Compare op slice info
  auto real_split_info_str = GetFusionOpSliceInfoStrFromGraph(compute_graph_ptr);
  auto expect_split_inf_str = "";
  EXPECT_EQ(real_split_info_str, expect_split_inf_str);
}

TEST_F(BatchMatMulElementwiseUbFusionTest, batch_matmul_mul_sigmoid_mul) {
  ge::Graph graph(this->test_info_->name());

  // step1: Construct Graph
  auto data_a = CreateData("data_a", ge::Shape({77, 32, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_b = CreateData("data_b", ge::Shape({128, 32, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);

  auto batch_matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                             .set_input_x1(data_a)
                             .set_input_x2(data_b)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);

  auto const_add = op::Const("scalar_for_mul0");
  Tensor const_add_tensor;
  float *const_add_tensor_value = new float[1];
  for (int i = 0; i < 1; i++) {
    *(const_add_tensor_value + i) = 1.7;
  }
  const_add_tensor.SetData((uint8_t *)const_add_tensor_value, 1 * 4);
  std::vector<int64_t> dims_add{1};
  ge::Shape shape_add(dims_add);
  ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT);
  const_add_tensor.SetTensorDesc(tensorDescAdd);
  const_add.set_attr_value(const_add_tensor);

  auto cast_op = op::Cast("input");                     // Note: op name used in construct BufferFusionMapping
  cast_op.set_input_x(const_add).set_attr_dst_type(1);  // dst_type 1: float16
  auto mul0 = op::Mul("mul0").set_input_x1(batch_matmul_op).set_input_x2(cast_op);

  auto sigmoid = op::Sigmoid("sigmoid").set_input_x(mul0);

  auto mul1 = op::Mul("mul1").set_input_x1(batch_matmul_op).set_input_x2(sigmoid);

  std::vector<Operator> inputs{data_a, data_b};
  std::vector<Operator> outputs{mul1};

  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  vector<AxisSplitMap> asm_mm{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})},  // batch
                         {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})},  // m
                         {CreateOutputSplitInfo(0, {2})}),
      CreateAxisSplitMap({CreateInputSplitInfo(1, {0})},  // n
                         {CreateOutputSplitInfo(0, {1})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"BatchMatMulV2"}));
  vector<AxisSplitMap> asm_mul0{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByName(compute_graph_ptr, asm_mul0, "mul0"));
  vector<AxisSplitMap> asm_sigmoid{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_sigmoid, {"Sigmoid"}));
  vector<AxisSplitMap> asm_mul1{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1}), CreateInputSplitInfo(1, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2}), CreateInputSplitInfo(1, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByName(compute_graph_ptr, asm_mul1, "mul1"));

  auto mapping = ConstructFusionMappingOfMulSigmoidMul(compute_graph_ptr);
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            compute_graph_ptr, mapping);
  EXPECT_EQ(res, fe::SUCCESS);

  auto real_split_info_str = GetFusionOpSliceInfoStrFromGraph(compute_graph_ptr);
  // Note: order as same as BatchMatMul
  auto expect_split_inf_str = CreateFusionOpSliceInfoStrFromSplitMap(
      {CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),
       CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),
       CreateAxisSplitMap({CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {1})})});
  EXPECT_EQ(real_split_info_str, expect_split_inf_str);
}

TEST_F(BatchMatMulElementwiseUbFusionTest, batch_matmul_elem_elem) {
  ge::Graph graph(this->test_info_->name());

  // step1: Construct Graph
  auto data_a = CreateData("data_a", ge::Shape({77, 32, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_b = CreateData("data_b", ge::Shape({128, 32, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);

  auto batch_matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                             .set_input_x1(data_a)
                             .set_input_x2(data_b)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);
  batch_matmul_op.update_output_desc_y(
      ge::TensorDesc(ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  auto data_c = CreateData("data_c", ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto add_0_op = op::Add("add_0").set_input_x1(batch_matmul_op).set_input_x2(data_c);
  add_0_op.update_output_desc_y(ge::TensorDesc(ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  auto data_d = CreateData("data_d", ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto add_1_op = op::Add("add_1").set_input_x1(add_0_op).set_input_x2(data_c);
  add_1_op.update_output_desc_y(ge::TensorDesc(ge::Shape({72, 128, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  std::vector<Operator> inputs{data_a, data_b};
  std::vector<Operator> outputs{add_1_op};

  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  vector<AxisSplitMap> asm_mm{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})},  // batch
                         {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})},  // m
                         {CreateOutputSplitInfo(0, {2})}),
      CreateAxisSplitMap({CreateInputSplitInfo(1, {0})},  // n
                         {CreateOutputSplitInfo(0, {1})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"BatchMatMulV2"}));
  vector<AxisSplitMap> asm_add_0{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1}), CreateInputSplitInfo(1, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2}), CreateInputSplitInfo(1, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByName(compute_graph_ptr, asm_add_0, "add_0"));
  EXPECT_TRUE(SetSplitMapToNodeByName(compute_graph_ptr, asm_add_0, "add_1"));

  auto mapping = ConstructFusionMappingOfElemElem(compute_graph_ptr);
  Status res = fe::FusionPassTestUtils::RunBufferFusionPass(ptr_buffer_fusion_pass_func.get(), patterns,
                                                            compute_graph_ptr, mapping);
  EXPECT_EQ(res, fe::SUCCESS);

  auto real_split_info_str = GetFusionOpSliceInfoStrFromGraph(compute_graph_ptr);
  // Note: order as same as BatchMatMul
  auto expect_split_inf_str = CreateFusionOpSliceInfoStrFromSplitMap(
      {CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(2, {0}), CreateInputSplitInfo(3, {0})},
                          {CreateOutputSplitInfo(0, {0})}),
       CreateAxisSplitMap({CreateInputSplitInfo(0, {2}), CreateInputSplitInfo(2, {2}), CreateInputSplitInfo(3, {2})},
                          {CreateOutputSplitInfo(0, {2})}),
       CreateAxisSplitMap({CreateInputSplitInfo(1, {0}), CreateInputSplitInfo(2, {1}), CreateInputSplitInfo(3, {1})},
                          {CreateOutputSplitInfo(0, {1})})});
  EXPECT_EQ(real_split_info_str, expect_split_inf_str);
}
