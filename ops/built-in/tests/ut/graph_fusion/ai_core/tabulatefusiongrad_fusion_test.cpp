/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "selection_ops.h"
#include "split_combination_ops.h"
#include "deep_md.h"
#include "fusion_pass_test_utils.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class tabulatefusiongrad_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tabulatefusiongrad_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tabulatefusiongrad_fusion_test TearDown" << std::endl;
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

TEST_F(tabulatefusiongrad_fusion_test, tabulatefusiongrad_fusion_test_01) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  platformInfo.str_info.short_soc_version = "Ascend310P";
  optiCompilationInfo.soc_version = "Ascend310P3";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310P3"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  std::string testCaseName = "tabulatefusiongrad_fusion_test_01";
  int32_t nloc = 12288;
  int32_t nnei = 138;
  int32_t lastLayerSize = 128;
  int32_t split_count = 1;
  int32_t split_index = 0;

  int32_t tableDim0 = 1024;

  auto table = CreateDataNode("table", {tableDim0, lastLayerSize * 6}, FORMAT_ND, DT_FLOAT);
  auto table_info = CreateDataNode("table_info", {6,}, FORMAT_ND, DT_FLOAT);
  auto em_x = CreateDataNode("em_x", {nloc, nnei}, FORMAT_ND, DT_FLOAT);
  auto em = CreateDataNode("em", {nloc, nnei, 4}, FORMAT_ND, DT_FLOAT);
  auto dy = CreateDataNode("dy", {nloc, 4, lastLayerSize}, FORMAT_ND, DT_FLOAT);
  auto descriptor = CreateDataNode("descriptor", {nloc, 4, lastLayerSize}, FORMAT_ND, DT_FLOAT);

  std::string opName = "TabulateFusionGrad_01";
  auto op = TabulateFusionGrad(opName.c_str());
  op.set_input_table(table)
      .set_input_table_info(table_info)
      .set_input_em_x(em_x)
      .set_input_em(em)
      .set_input_dy(dy)
      .set_input_descriptor(descriptor);
  op.update_output_desc_dy_dem_x(SimpleTensorDesc("dy_dem_x", {nloc, nnei}, FORMAT_ND, DT_FLOAT));
  op.update_output_desc_dy_dem(SimpleTensorDesc("dy_dem", {nloc, nnei, 4}, FORMAT_ND, DT_FLOAT));

  std::vector<Operator> inputs{table, table_info, em_x, em, dy, descriptor};
  std::vector<Operator> outputs{op};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("TabulateFusionGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findAiCoreNode = false;
  bool findVectorCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "TabulateFusionGrad") {
      ge::OpDescPtr iOpDesc = iNode->GetOpDesc();

      std::string engineName;
      bool engineRet = ge::AttrUtils::GetStr(iOpDesc, "_specified_engine_name", engineName);
      EXPECT_EQ(engineRet, true);

      std::string kernelLibName;
      bool kernelLibRet = ge::AttrUtils::GetStr(iOpDesc, "_specified_kernel_lib_name", kernelLibName);
      EXPECT_EQ(kernelLibRet, true);

      if (engineName == "AIcoreEngine" && kernelLibName == "AIcoreEngine") {
        findAiCoreNode = true;
      } else if (engineName == "VectorEngine" && kernelLibName == "VectorEngine") {
        findVectorCoreNode = true;
      }
    }
  }
  EXPECT_EQ(findAiCoreNode, true);
  EXPECT_EQ(findVectorCoreNode, true);

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}

TEST_F(tabulatefusiongrad_fusion_test, tabulatefusiongrad_fusion_test_02) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  platformInfo.str_info.short_soc_version = "Ascend310P";
  optiCompilationInfo.soc_version = "Ascend310P3";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310P3"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  std::string testCaseName = "tabulatefusiongrad_fusion_test_02";
  int32_t nloc = -1;
  int32_t nnei = 138;
  int32_t lastLayerSize = 128;
  int32_t split_count = 1;
  int32_t split_index = 0;

  int32_t tableDim0 = 1024;

  auto table = CreateDataNode("table", {tableDim0, lastLayerSize * 6}, FORMAT_ND, DT_FLOAT);
  auto table_info = CreateDataNode("table_info", {6,}, FORMAT_ND, DT_FLOAT);
  auto em_x = CreateDataNode("em_x", {nloc, nnei}, FORMAT_ND, DT_FLOAT);
  auto em = CreateDataNode("em", {nloc, nnei, 4}, FORMAT_ND, DT_FLOAT);
  auto dy = CreateDataNode("dy", {nloc, 4, lastLayerSize}, FORMAT_ND, DT_FLOAT);
  auto descriptor = CreateDataNode("descriptor", {nloc, 4, lastLayerSize}, FORMAT_ND, DT_FLOAT);

  std::string opName = "TabulateFusionGrad_02";
  auto op = TabulateFusionGrad(opName.c_str());
  op.set_input_table(table)
      .set_input_table_info(table_info)
      .set_input_em_x(em_x)
      .set_input_em(em)
      .set_input_dy(dy)
      .set_input_descriptor(descriptor);
  op.update_output_desc_dy_dem_x(SimpleTensorDesc("dy_dem_x", {nloc, nnei}, FORMAT_ND, DT_FLOAT));
  op.update_output_desc_dy_dem(SimpleTensorDesc("dy_dem", {nloc, nnei, 4}, FORMAT_ND, DT_FLOAT));

  std::vector<Operator> inputs{table, table_info, em_x, em, dy, descriptor};
  std::vector<Operator> outputs{op};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("TabulateFusionGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findAiCoreNode = false;
  bool findVectorCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "TabulateFusionGrad") {
      ge::OpDescPtr iOpDesc = iNode->GetOpDesc();

      std::string engineName;
      bool engineRet = ge::AttrUtils::GetStr(iOpDesc, "_specified_engine_name", engineName);
      EXPECT_EQ(engineRet, true);

      std::string kernelLibName;
      bool kernelLibRet = ge::AttrUtils::GetStr(iOpDesc, "_specified_kernel_lib_name", kernelLibName);
      EXPECT_EQ(kernelLibRet, true);

      if (engineName == "AIcoreEngine" && kernelLibName == "AIcoreEngine") {
        findAiCoreNode = true;
      } else if (engineName == "VectorEngine" && kernelLibName == "VectorEngine") {
        findVectorCoreNode = true;
      }
    }
  }
  EXPECT_EQ(findAiCoreNode, true);
  EXPECT_EQ(findVectorCoreNode, true);

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}
