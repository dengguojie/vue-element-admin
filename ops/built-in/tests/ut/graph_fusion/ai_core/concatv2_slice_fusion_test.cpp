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
#include "deep_md.h"
#include "elewise_calculation_ops.h"
#include "pad_ops.h"
#include "selection_ops.h"
#include "split_combination_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph_builder_utils.h"

using namespace ge;
using namespace op;

class concatv2_slice_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "concatv2_slice_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "concatv2_slice_fusion_test TearDown" << std::endl;
  }
};


TEST_F(concatv2_slice_fusion_test, concatv2_slice_fusion_test_01) {
  std::cout << "concatv2_slice_fusion_test.concatv2_slice_fusion_test_01 start." << std::endl;

  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());

  auto dataNode1 = builder.AddNode("Data1", "Data", 0, 1, {1, 125}, FORMAT_ND, DT_FLOAT);
  auto dataNode2 = builder.AddNode("Data2", "Data", 0, 1, {1, 128}, FORMAT_ND, DT_FLOAT);
  auto constantNode1 = builder.AddNode("constant1", "Constant", 0, 1, {4748, 1}, FORMAT_ND, DT_FLOAT);

  auto concatV2Node0 = builder.AddNode(
      "ConcatV20", "ConcatV2", {{Format::FORMAT_ND, DT_FLOAT, {1, 125}}, {Format::FORMAT_ND, DT_FLOAT, {1, 128}}},
      {{Format::FORMAT_ND, DT_FLOAT, {1, 253}}});
  auto sliceNode0 = builder.AddNode(
      "Slice0", "Slice", {{Format::FORMAT_ND, DT_FLOAT, {1, 253}}},
      {{Format::FORMAT_ND, DT_FLOAT, {1, 125}}});
  auto sliceNode1 = builder.AddNode(
      "Slice1", "Slice", {{Format::FORMAT_ND, DT_FLOAT, {1, 253}}},
      {{Format::FORMAT_ND, DT_FLOAT, {1, 128}}});
  auto concatV2Node1 = builder.AddNode(
      "ConcatV21", "ConcatV2", {{Format::FORMAT_ND, DT_FLOAT, {1, 125}}, {Format::FORMAT_ND, DT_FLOAT, {1, 128}}},
      {{Format::FORMAT_ND, DT_FLOAT, {1, 253}}});
  auto netOutput = builder.AddNode("NetOutput", "NetOutput", 1, 0, {1, 253}, FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(dataNode1, 0, concatV2Node0, 0);
  builder.AddDataEdge(dataNode2, 0, concatV2Node0, 1);
  builder.AddDataEdge(concatV2Node0, 0, sliceNode0, 0);
  builder.AddDataEdge(concatV2Node0, 0, sliceNode1, 0);
  builder.AddDataEdge(sliceNode0, 0, concatV2Node1, 0);
  builder.AddDataEdge(sliceNode1, 0, concatV2Node1, 1);
  builder.AddDataEdge(concatV2Node1, 0, netOutput, 0);

  ge::AttrUtils::SetInt(concatV2Node0->GetOpDesc(), "N", 2);
  GeTensorDesc concatDimDesc(ge::GeShape(std::vector<int64_t>({})), ge::FORMAT_ND, DT_INT32);
  int32_t concatDimData = 1;
  auto concatDimTensor =
      std::make_shared<ge::GeTensor>(concatDimDesc, reinterpret_cast<uint8_t *>(&concatDimData), sizeof(concatDimData));
  OpDescUtils::SetWeights(concatV2Node0, {concatDimTensor});
  concatV2Node0->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}, {"concat_dim", 2}});

  OpDescUtils::SetWeights(concatV2Node1, {concatDimTensor});
  concatV2Node1->GetOpDesc()->UpdateInputName({{"x0", 0}, {"x1", 1}, {"concat_dim", 2}});

  vector<int32_t> offset0Vec = {0, 0};
  GeTensorDesc offset0Desc(ge::GeShape(std::vector<int64_t>({2})), ge::FORMAT_ND, DT_INT32);
  auto offset0Tensor = std::make_shared<GeTensor>(offset0Desc,
                                        reinterpret_cast<uint8_t *>(offset0Vec.data()),
                                        offset0Vec.size() * sizeof(DT_INT32));
  vector<int32_t> size0Vec = {1, 125};
  GeTensorDesc size0Desc(ge::GeShape(std::vector<int64_t>({2})), ge::FORMAT_ND, DT_INT32);
  auto size0Tensor = std::make_shared<GeTensor>(size0Desc,
                                        reinterpret_cast<uint8_t *>(size0Vec.data()),
                                        size0Vec.size() * sizeof(DT_INT32));
  std::map<int, ge::GeTensorPtr> weights_map0;
  weights_map0[1] = offset0Tensor;
  weights_map0[2] = size0Tensor;
  ge::OpDescUtils::SetWeights(*sliceNode0, weights_map0);
  sliceNode0->GetOpDesc()->UpdateInputName({{"x", 0}, {"offsets", 1}, {"size", 2}});

  vector<int32_t> offset1Vec = {0, 125};
  GeTensorDesc offset1Desc(ge::GeShape(std::vector<int64_t>({2})), ge::FORMAT_ND, DT_INT32);
  auto offset1Tensor = std::make_shared<GeTensor>(offset1Desc,
                                        reinterpret_cast<uint8_t *>(offset1Vec.data()),
                                        offset1Vec.size() * sizeof(DT_INT32));
  vector<int32_t> size1Vec = {1, 128};
  GeTensorDesc size1Desc(ge::GeShape(std::vector<int64_t>({2})), ge::FORMAT_ND, DT_INT32);
  auto size1Tensor = std::make_shared<GeTensor>(size1Desc,
                                        reinterpret_cast<uint8_t *>(size1Vec.data()),
                                        size1Vec.size() * sizeof(DT_INT32));
  std::map<int, ge::GeTensorPtr> weights_map1;
  weights_map1[1] = offset1Tensor;
  weights_map1[2] = size1Tensor;
  ge::OpDescUtils::SetWeights(*sliceNode1, weights_map1);
  sliceNode1->GetOpDesc()->UpdateInputName({{"x", 0}, {"offsets", 1}, {"size", 2}});

  auto graph = builder.GetGraph();

  GraphUtils::DumpGEGraphToOnnx(*graph, "0");
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("ASliceConcatV2FusionPass",
                                                         fe::BUILT_IN_GRAPH_PASS,
                                                         *graph);
  int op_count = 0;
  GraphUtils::DumpGEGraphToOnnx(*graph, "1");
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "ConcatV2") {
      op_count++;
    }

    if (node->GetType() == "Slice") {
      op_count++;
    }
  }
  EXPECT_EQ(op_count, 1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::cout << "concatv2_slice_fusion_test.concatv2_slice_fusion_test_01 successful." << std::endl;
}