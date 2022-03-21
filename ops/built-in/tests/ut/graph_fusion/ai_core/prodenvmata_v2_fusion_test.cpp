/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "deep_md.h"
#define private public
#define protected public
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;
using namespace std;

class prodenvmata_v2_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "prodenvmata_v2_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "prodenvmata_v2_fusion_test TearDown" << std::endl;
  }
};

#define DESC_DATA(name, shape_in, format_in, shape_out, format_out, dtype) \
  ge::GeTensorDesc desc_##name(shape_out, format_out, dtype);              \
  desc_##name.SetOriginFormat(format_in);                                  \
  desc_##name.SetOriginShape(shape_in)

TEST_F(prodenvmata_v2_fusion_test, prodenvmata_v2_fusion_test_1) {
  ge::Graph graph("prodenvmata_v2_fusion_test_1");

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 32;
  optiCompilationInfo.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  DESC_DATA(coord, ge::GeShape({1, 4608}), FORMAT_ND, ge::GeShape({1, 4608}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(type, ge::GeShape({1, 1536}), FORMAT_ND, ge::GeShape({1, 1536}), FORMAT_ND, DT_INT32);
  DESC_DATA(natoms, ge::GeShape({3}), FORMAT_ND, ge::GeShape({3}), FORMAT_ND, DT_INT32);
  DESC_DATA(box, ge::GeShape({1, 576}), FORMAT_ND, ge::GeShape({1, 576}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(mesh, ge::GeShape({198529}), FORMAT_ND, ge::GeShape({198529}), FORMAT_ND, DT_INT32);
  DESC_DATA(davg, ge::GeShape({2, 552}), FORMAT_ND, ge::GeShape({2, 552}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(dstd, ge::GeShape({2, 552}), FORMAT_ND, ge::GeShape({2, 552}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(prodEnvMatA, ge::GeShape({1, 552}), FORMAT_ND, ge::GeShape({1, 552}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(prodOut0, ge::GeShape({1, 552}), FORMAT_ND, ge::GeShape({1, 552}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(prodOut1, ge::GeShape({1, 1656}), FORMAT_ND, ge::GeShape({1, 1656}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(prodOut2, ge::GeShape({1, 414}), FORMAT_ND, ge::GeShape({1, 414}), FORMAT_ND, DT_FLOAT);
  DESC_DATA(prodOut3, ge::GeShape({1, 138}), FORMAT_ND, ge::GeShape({1, 138}), FORMAT_ND, DT_INT32);

  ge::OpDescPtr coord = std::make_shared<ge::OpDesc>("coord", "Data");
  ge::OpDescPtr type = std::make_shared<ge::OpDesc>("type", "Data");
  ge::OpDescPtr natoms = std::make_shared<ge::OpDesc>("natoms", "Data");
  ge::OpDescPtr box = std::make_shared<ge::OpDesc>("box", "Data");
  ge::OpDescPtr mesh = std::make_shared<ge::OpDesc>("mesh", "Data");
  ge::OpDescPtr davg = std::make_shared<ge::OpDesc>("davg", "Data");
  ge::OpDescPtr dstd = std::make_shared<ge::OpDesc>("dstd", "Data");
  ge::OpDescPtr prodEnvMatA = std::make_shared<ge::OpDesc>("prodEnvMatA", "ProdEnvMatA");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("output", "NetOutput");

  coord->AddOutputDesc(desc_coord);
  type->AddOutputDesc(desc_type);
  natoms->AddOutputDesc(desc_natoms);
  box->AddOutputDesc(desc_box);
  mesh->AddOutputDesc(desc_mesh);
  davg->AddOutputDesc(desc_davg);
  dstd->AddOutputDesc(desc_dstd);

  prodEnvMatA->AddInputDesc("coord", desc_coord);
  prodEnvMatA->AddInputDesc("type", desc_type);
  prodEnvMatA->AddInputDesc("natoms", desc_natoms);
  prodEnvMatA->AddInputDesc("box", desc_mesh);
  prodEnvMatA->AddInputDesc("mesh", desc_davg);
  prodEnvMatA->AddInputDesc("davg", desc_coord);
  prodEnvMatA->AddInputDesc("dstd", desc_dstd);
  prodEnvMatA->AddOutputDesc("dstd", desc_prodOut0);
  prodEnvMatA->AddOutputDesc("descrpt", desc_prodOut1);
  prodEnvMatA->AddOutputDesc("rij", desc_prodOut2);
  prodEnvMatA->AddOutputDesc("nlist", desc_prodOut3);

  ge::AttrUtils::SetFloat(prodEnvMatA, "rcut_a", 0.0);
  ge::AttrUtils::SetFloat(prodEnvMatA, "rcut_r", 6.0);
  ge::AttrUtils::SetFloat(prodEnvMatA, "rcut_r_smth", 0.5);
  ge::AttrUtils::SetListInt(prodEnvMatA, "sel_a", {46, 92});
  ge::AttrUtils::SetListInt(prodEnvMatA, "sel_r", {0, 0});
  ge::AttrUtils::SetInt(prodEnvMatA, "split_count", 1);
  ge::AttrUtils::SetInt(prodEnvMatA, "split_index", 0);

  netoutput->AddInputDesc(desc_prodOut0);
  netoutput->AddInputDesc(desc_prodOut1);
  netoutput->AddInputDesc(desc_prodOut2);
  netoutput->AddInputDesc(desc_prodOut3);

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("ProdEnvMatAV2FusionPass_graph");
  ge::NodePtr coord_node = compute_graph_ptr->AddNode(coord);
  ge::NodePtr type_node = compute_graph_ptr->AddNode(type);
  ge::NodePtr natoms_node = compute_graph_ptr->AddNode(natoms);
  ge::NodePtr box_node = compute_graph_ptr->AddNode(box);
  ge::NodePtr mesh_node = compute_graph_ptr->AddNode(mesh);
  ge::NodePtr davg_node = compute_graph_ptr->AddNode(davg);
  ge::NodePtr dstd_node = compute_graph_ptr->AddNode(dstd);
  ge::NodePtr prodEnvMatA_node = compute_graph_ptr->AddNode(prodEnvMatA);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);

  ge::GraphUtils::AddEdge(coord_node->GetOutDataAnchor(0), prodEnvMatA_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(type_node->GetOutDataAnchor(0), prodEnvMatA_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(natoms_node->GetOutDataAnchor(0), prodEnvMatA_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(box_node->GetOutDataAnchor(0), prodEnvMatA_node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(mesh_node->GetOutDataAnchor(0), prodEnvMatA_node->GetInDataAnchor(4));
  ge::GraphUtils::AddEdge(davg_node->GetOutDataAnchor(0), prodEnvMatA_node->GetInDataAnchor(5));
  ge::GraphUtils::AddEdge(dstd_node->GetOutDataAnchor(0), prodEnvMatA_node->GetInDataAnchor(6));
  ge::GraphUtils::AddEdge(prodEnvMatA_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(prodEnvMatA_node->GetOutDataAnchor(1), netoutput_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(prodEnvMatA_node->GetOutDataAnchor(2), netoutput_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(prodEnvMatA_node->GetOutDataAnchor(3), netoutput_node->GetInDataAnchor(3));

  fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("ProdEnvMatAV2FusionPass", fe::BUILT_IN_GRAPH_PASS,
                                                                  *compute_graph_ptr);

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  EXPECT_EQ(status, fe::SUCCESS);
}