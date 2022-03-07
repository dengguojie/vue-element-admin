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
#include "pad_ops.h"
#include "deep_md.h"
#include "fusion_pass_test_utils.h"
#define private public
#include "common/util/platform_info.h"

using namespace std;
using namespace ge;
using namespace op;

class padd_update_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "padd_update_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "padd_update_fusion_test TearDown" << std::endl;
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

TEST_F(padd_update_fusion_test, padd_update_fusion_test_01) {
  std::string testCaseName = "padd_update_fusion_test_01";

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  optiCompilationInfo.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  auto x = CreateDataNode("x", {1, 6553600}, FORMAT_ND, DT_FLOAT);

  auto paddOp = PadD("PadD_01");
  paddOp.set_input_x(x).set_attr_paddings({{2, 2}, {2, 2}});
  paddOp.update_output_desc_y(SimpleTensorDesc("y", {5, 6553604}, FORMAT_ND, DT_FLOAT));

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{paddOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("PaddUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findPaddNode = false;
  bool findPadNode = false;
  bool findConstNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "PadD") {
      findPaddNode = true;
    } else if (iNode->GetType() == "Pad") {
      findPadNode = true;
    } else if (iNode->GetType() == "Const") {
      findConstNode = true;
    }
  }

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  EXPECT_EQ(findPaddNode, false);
  EXPECT_EQ(findPadNode, true);
  EXPECT_EQ(findConstNode, true);
}

TEST_F(padd_update_fusion_test, padd_update_fusion_test_02) {
  std::string testCaseName = "padd_update_fusion_test_02";

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  optiCompilationInfo.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  auto x = CreateDataNode("x", {1, 16, 3, 7, 16}, FORMAT_NC1HWC0, DT_FLOAT);

  auto paddOp = PadD("PadD_02");
  paddOp.set_input_x(x).set_attr_paddings({{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}});
  paddOp.update_output_desc_y(SimpleTensorDesc("y", {5, 20, 7, 11, 20}, FORMAT_NC1HWC0, DT_FLOAT));

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{paddOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("PaddUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findPaddNode = false;
  bool findPadNode = false;
  bool findConstNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "PadD") {
      findPaddNode = true;
    } else if (iNode->GetType() == "Pad") {
      findPadNode = true;
    } else if (iNode->GetType() == "Const") {
      findConstNode = true;
    }
  }

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  EXPECT_EQ(findPaddNode, true);
  EXPECT_EQ(findPadNode, false);
  EXPECT_EQ(findConstNode, false);
}

TEST_F(padd_update_fusion_test, padd_update_fusion_test_03) {
  std::string testCaseName = "padd_update_fusion_test_03";

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  optiCompilationInfo.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  auto x = CreateDataNode("x", {1, 16, 3, 7, 16}, FORMAT_NC1HWC0, DT_FLOAT);

  auto paddOp = PadD("PadD_03");
  paddOp.set_input_x(x).set_attr_paddings({{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
  paddOp.update_output_desc_y(SimpleTensorDesc("y", {1, 16, 3, 7, 16}, FORMAT_NC1HWC0, DT_FLOAT));

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{paddOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("PaddUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findPaddNode = false;
  bool findPadNode = false;
  bool findConstNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "PadD") {
      findPaddNode = true;
    } else if (iNode->GetType() == "Pad") {
      findPadNode = true;
    } else if (iNode->GetType() == "Const") {
      findConstNode = true;
    }
  }

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  EXPECT_EQ(findPaddNode, true);
  EXPECT_EQ(findPadNode, false);
  EXPECT_EQ(findConstNode, false);
}

TEST_F(padd_update_fusion_test, padd_update_fusion_test_04) {
  std::string testCaseName = "padd_update_fusion_test_04";

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  optiCompilationInfo.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  auto x = CreateDataNode("x", {1, 3200, 256}, FORMAT_ND, DT_FLOAT);

  auto paddOp = PadD("PadD_04");
  paddOp.set_input_x(x).set_attr_paddings({{0, 0}, {2, 2}, {0, 0}});
  paddOp.update_output_desc_y(SimpleTensorDesc("y", {1, 3204, 256}, FORMAT_ND, DT_FLOAT));

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{paddOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("PaddUpdateFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findPaddNode = false;
  bool findPadNode = false;
  bool findConstNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "PadD") {
      findPaddNode = true;
    } else if (iNode->GetType() == "Pad") {
      findPadNode = true;
    } else if (iNode->GetType() == "Const") {
      findConstNode = true;
    }
  }

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  EXPECT_EQ(findPaddNode, true);
  EXPECT_EQ(findPadNode, false);
  EXPECT_EQ(findConstNode, false);
}
