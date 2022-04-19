/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include <stdlib.h>
#include <nlohmann/json.hpp>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "split_combination_ops.h"
#include "transformation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class relu_cast_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "relu_cast_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "relu_cast_fusion_test TearDown" << std::endl;
  }
};

TensorDesc SimpleTensorDesc(std::string name, std::vector<int64_t> dims, Format format, DataType dataType) {
  ge::Shape shape0(dims);

  TensorDesc tensorDesc(shape0, format, dataType);
  tensorDesc.SetName(name.c_str());
  tensorDesc.SetOriginShape(shape0);
  tensorDesc.SetOriginFormat(format);

  return tensorDesc;
}

Data CreateDataNode(const std::string& nodeName, const std::vector<int64_t>& dims, const Format& format,
                    const DataType& dataType) {
  Data data = Data(nodeName.c_str());
  data.update_input_desc_x(SimpleTensorDesc(nodeName, dims, format, dataType));
  data.update_output_desc_y(SimpleTensorDesc(nodeName, dims, format, dataType));
  return data;
}

TEST_F(relu_cast_fusion_test, relu_cast_fusion_test_01) {
  std::string testCaseName = "relu_cast_fusion_test_01";

  auto x0 = CreateDataNode("x0", {640, 110}, FORMAT_ND, DT_FLOAT);
  auto x1 = CreateDataNode("x1", {640, 4510}, FORMAT_ND, DT_FLOAT);

  auto concatOp = ConcatV2D("ConcatV2D_11");
  concatOp.create_dynamic_input_x(2)
      .set_dynamic_input_x(0, x0)
      .set_dynamic_input_x(1, x1)
      .set_attr_concat_dim(1)
      .set_attr_N(2);
  concatOp.update_output_desc_y(SimpleTensorDesc("y", {640, 4620}, FORMAT_ND, DT_FLOAT));

  auto transdata = op::TransData("TransData_01")
      .set_input_src(concatOp)
      .set_attr_src_format("ND")
      .set_attr_dst_format("FRACTAL_NZ");
  transdata.update_input_desc_src(SimpleTensorDesc("src", {640, 4620}, FORMAT_ND, DT_FLOAT));
  transdata.update_output_desc_dst(SimpleTensorDesc("dst", {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT));

  auto cast1 = Cast("Cast_01");
  cast1.set_input_x(transdata).set_attr_dst_type(DT_FLOAT16);

  auto relu = Relu("Relu_01");
  relu.set_input_x(transdata);

  auto cast2 = Cast("Cast_02");
  cast2.set_input_x(relu).set_attr_dst_type(DT_FLOAT16);

  auto endOp = Maximum("Maximum_01");
  endOp.set_input_x1(cast1).set_input_x2(cast2);

  std::vector<Operator> inputs{x0, x1};
  std::vector<Operator> outputs{endOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ReluCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  ge::NodePtr transDataNode;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "TransData") {
      transDataNode = iNode;
      break;
    }
  }
  EXPECT_EQ(transDataNode != nullptr, true);
  EXPECT_EQ(transDataNode->GetOutDataNodesSize() == 2, true);

  for (auto castNode : transDataNode->GetOutDataNodes()) {
    EXPECT_EQ(castNode->GetType() == "Cast", true);
    if (castNode->GetName() == "Cast_02") {
      auto reluNode = castNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(reluNode->GetType() == "Relu", true);

      ge::OpDescPtr reluDesc = reluNode->GetOpDesc();
      EXPECT_EQ(reluDesc->GetInputDesc(0).GetDataType() == DT_FLOAT16, true);
      EXPECT_EQ(reluDesc->GetOutputDesc(0).GetDataType() == DT_FLOAT16, true);
    }
  }
}

TEST_F(relu_cast_fusion_test, relu_cast_fusion_test_02) {
  std::string testCaseName = "relu_cast_fusion_test_02";

  auto data = CreateDataNode("data_21", {640, 4620}, FORMAT_ND, DT_FLOAT);

  auto transdata = op::TransData("TransData_21")
      .set_input_src(data)
      .set_attr_src_format("ND")
      .set_attr_dst_format("FRACTAL_NZ");
  transdata.update_input_desc_src(SimpleTensorDesc("src", {640, 4620}, FORMAT_ND, DT_FLOAT));
  transdata.update_output_desc_dst(SimpleTensorDesc("dst", {289, 40, 16, 16}, FORMAT_FRACTAL_NZ, DT_FLOAT));

  auto relu = Relu("Relu_21");
  relu.set_input_x(transdata);

  auto cast = Cast("Cast_21");
  cast.set_input_x(relu).set_attr_dst_type(DT_FLOAT16);

  auto endOp = Maximum("Maximum_21");
  endOp.set_input_x1(cast).set_input_x2(cast);

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{endOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ReluCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  ge::NodePtr transDataNode;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "TransData") {
      transDataNode = iNode;
      break;
    }
  }
  EXPECT_EQ(transDataNode != nullptr, true);
  EXPECT_EQ(transDataNode->GetOutDataNodes().at(0)->GetType() == "Relu", true);
}
