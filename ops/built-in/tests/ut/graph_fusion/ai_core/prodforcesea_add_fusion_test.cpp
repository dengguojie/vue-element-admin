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

class prodforcesea_add_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "prodforcesea_add_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "prodforcesea_add_fusion_test TearDown" << std::endl;
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

TEST_F(prodforcesea_add_fusion_test, prodforcesea_add_fusion_test_01) {
  std::string testCaseName = "prodforcesea_add_fusion_test_01";
  int32_t nframes = 1;
  int32_t nloc = 12288;
  int32_t nall = 28328;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;
  int32_t natomsSize = 4;

  auto netDeriv = CreateDataNode("net_deriv", {nframes, nloc * nnei * 4}, FORMAT_ND, DT_FLOAT);
  auto inDeriv = CreateDataNode("in_deriv", {nframes, nloc * nnei * 4 * 3}, FORMAT_ND, DT_FLOAT);
  auto nlist = CreateDataNode("nlist", {nframes, nloc * nnei}, FORMAT_ND, DT_INT32);

  auto natoms = ge::op::Const("natoms");
  {
    ge::TensorDesc tensorDesc(ge::Shape({natomsSize}), ge::FORMAT_ND, ge::DT_INT32);
    int32_t tensorValue[natomsSize] = {nloc, nall, 0, 1};
    ge::Tensor natomsTensor = ge::Tensor(tensorDesc, (uint8_t*)tensorValue, natomsSize * sizeof(int32_t));

    natoms.UpdateOutputDesc("y", tensorDesc);
    natoms.set_attr_value(natomsTensor);
  }

  std::string forceOpName = "ProdForceSeA_01";
  auto forceOp = ProdForceSeA(forceOpName.c_str());
  forceOp.set_input_net_deriv(netDeriv)
      .set_input_in_deriv(inDeriv)
      .set_input_nlist(nlist)
      .set_input_natoms(natoms)
      .set_attr_n_a_sel(n_a_sel)
      .set_attr_n_r_sel(n_r_sel);
  forceOp.update_output_desc_atom_force(SimpleTensorDesc("force", {nframes, nall, 3}, FORMAT_ND, DT_FLOAT));

  std::vector<Operator> inputs{netDeriv, inDeriv, nlist, natoms};
  std::vector<Operator> outputs{forceOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  platformInfo.str_info.short_soc_version = "Ascend310P";
  optiCompilationInfo.soc_version = "Ascend310P3";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310P3"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ProdForceSeAVectorFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  bool findAiCoreNode = false;
  bool findVectorCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "ProdForceSeA") {
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
}

TEST_F(prodforcesea_add_fusion_test, prodforcesea_add_fusion_test_02) {
  std::string testCaseName = "prodforcesea_add_fusion_test_02";
  int32_t nframes = 1;
  int32_t nloc = -1;
  int32_t nall = -1;
  int32_t n_a_sel = 138;
  int32_t n_r_sel = 0;
  int32_t nnei = n_a_sel + n_r_sel;

  auto netDeriv = CreateDataNode("net_deriv", {nframes, -1}, FORMAT_ND, DT_FLOAT);
  auto inDeriv = CreateDataNode("in_deriv", {nframes, -1}, FORMAT_ND, DT_FLOAT);
  auto nlist = CreateDataNode("nlist", {nframes, -1}, FORMAT_ND, DT_INT32);
  auto natoms = CreateDataNode("natoms", {4}, FORMAT_ND, DT_INT32);

  std::string forceOpName = "ProdForceSeA_02";
  auto forceOp = ProdForceSeA(forceOpName.c_str());
  forceOp.set_input_net_deriv(netDeriv)
      .set_input_in_deriv(inDeriv)
      .set_input_nlist(nlist)
      .set_input_natoms(natoms)
      .set_attr_n_a_sel(n_a_sel)
      .set_attr_n_r_sel(n_r_sel);
  forceOp.update_output_desc_atom_force(SimpleTensorDesc("force", {nframes, nall, 3}, FORMAT_ND, DT_FLOAT));

  std::vector<Operator> inputs{netDeriv, inDeriv, nlist, natoms};
  std::vector<Operator> outputs{forceOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  platformInfo.str_info.short_soc_version = "Ascend310P";
  optiCompilationInfo.soc_version = "Ascend310P3";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310P3"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ProdForceSeAVectorFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  bool findAiCoreNode = false;
  bool findVectorCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "ProdForceSeA") {
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
}
