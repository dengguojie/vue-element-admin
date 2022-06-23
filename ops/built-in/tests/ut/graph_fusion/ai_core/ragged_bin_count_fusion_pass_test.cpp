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

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"
#include "elewise_calculation_ops.h"
#include "math_ops.h"
#include "array_ops.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;
using namespace fe;

class RaggedBinCountFusionPassTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RaggedBinCountFusionPassTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RaggedBinCountFusionPassTest TearDown" << std::endl;
  }

  ge::NodePtr GetNode(ComputeGraphPtr& graph, const string& name) {
    for (auto& node : graph->GetDirectNode()) {
      if (node->GetName() == name) {
        return node;
      }
    }
    return nullptr;
  }

  ComputeGraphPtr CreateComplexGraph() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("rbc_fusion_pass_test_case");

    ge::OpDescPtr op_desc_rbc = std::make_shared<OpDesc>("ragged_bin_count", "RaggedBinCount");
    ge::OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    ge::OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");

    // add descriptor
    vector<int64_t> dim_splits = {6};
    ge::GeShape shape_splits(dim_splits);
    ge::GeTensorDesc tensor_desc_splits(shape_splits);
    tensor_desc_splits.SetFormat(FORMAT_ND);
    tensor_desc_splits.SetOriginFormat(FORMAT_ND);
    tensor_desc_splits.SetDataType(DT_INT64);
    tensor_desc_splits.SetDataType(DT_INT64);

    vector<int64_t> dim_values = {10};
    ge::GeShape shape_values(dim_values);
    ge::GeTensorDesc tensor_desc_values(shape_values);
    tensor_desc_values.SetFormat(FORMAT_ND);
    tensor_desc_values.SetOriginFormat(FORMAT_ND);
    tensor_desc_values.SetDataType(DT_INT32);
    tensor_desc_values.SetDataType(DT_INT32);

    vector<int64_t> dim_size = {1};
    ge::GeShape shape_size(dim_size);
    ge::GeTensorDesc tensor_desc_size(shape_size);
    tensor_desc_size.SetFormat(FORMAT_ND);
    tensor_desc_size.SetOriginFormat(FORMAT_ND);
    tensor_desc_size.SetDataType(DT_INT32);
    tensor_desc_size.SetDataType(DT_INT32);

    vector<int64_t> dim_weights = {10};
    ge::GeShape shape_weights(dim_weights);
    ge::GeTensorDesc tensor_desc_weights(shape_weights);
    tensor_desc_weights.SetFormat(FORMAT_ND);
    tensor_desc_weights.SetOriginFormat(FORMAT_ND);
    tensor_desc_weights.SetDataType(DT_FLOAT);
    tensor_desc_weights.SetDataType(DT_FLOAT);

    vector<int64_t> dim_output = {5, 5};
    ge::GeShape shape_output(dim_output);
    ge::GeTensorDesc tensor_desc_output(shape_output);
    tensor_desc_output.SetFormat(FORMAT_ND);
    tensor_desc_output.SetOriginFormat(FORMAT_ND);
    tensor_desc_output.SetDataType(DT_FLOAT);
    tensor_desc_output.SetDataType(DT_FLOAT);

    vector<int64_t> dim_x = {5, 5};
    ge::GeShape shape_x(dim_x);
    ge::GeTensorDesc tensor_desc_x(shape_x);
    tensor_desc_x.SetFormat(FORMAT_ND);
    tensor_desc_x.SetOriginFormat(FORMAT_ND);
    tensor_desc_x.SetDataType(DT_FLOAT);
    tensor_desc_x.SetDataType(DT_FLOAT);

    vector<int64_t> dim_y = {5, 5};
    ge::GeShape shape_y(dim_y);
    ge::GeTensorDesc tensor_desc_y(shape_y);
    tensor_desc_y.SetFormat(FORMAT_ND);
    tensor_desc_y.SetOriginFormat(FORMAT_ND);
    tensor_desc_y.SetDataType(DT_FLOAT);
    tensor_desc_y.SetDataType(DT_FLOAT);

    op_desc_rbc->AddInputDesc(tensor_desc_splits);
    op_desc_rbc->AddInputDesc(tensor_desc_values);
    op_desc_rbc->AddInputDesc(tensor_desc_size);
    op_desc_rbc->AddInputDesc(tensor_desc_weights);
    op_desc_rbc->AddOutputDesc(tensor_desc_output);

    op_desc_relu->AddInputDesc(tensor_desc_output);
    op_desc_relu->AddOutputDesc(tensor_desc_x);

    op_desc_output->AddInputDesc(tensor_desc_x);
    op_desc_output->AddOutputDesc(tensor_desc_y);

    NodePtr node_rbc = graph->AddNode(op_desc_rbc);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);

    GraphUtils::AddEdge(node_rbc->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    return graph;
  }
};

bool HasOutDataTemp(const ge::NodePtr& node) {
  FUSION_TURBO_NOTNULL(node, false);
  const auto out_data_anchors = node->GetAllOutDataAnchors();
  for (const auto& out_anchor : out_data_anchors) {
    for (const auto& peer_in_data_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_data_anchor->GetOwnerNode() != nullptr) {
        return true;
      }
    }
  }
  return false;
}

TEST_F(RaggedBinCountFusionPassTest, test_case_01) {
  auto graph = CreateComplexGraph();
  auto rbc_node = graph->FindFirstNodeMatchType("RaggedBinCount");
  bool has_data_out = HasOutDataTemp(rbc_node);
  EXPECT_EQ(has_data_out, true);
  has_data_out = HasOutDataTemp(nullptr);
  EXPECT_EQ(has_data_out, false);
  auto relu_node = graph->FindFirstNodeMatchType("Relu");
  ASSERT_NE(relu_node, nullptr);
  has_data_out = HasOutDataTemp(relu_node);
  EXPECT_EQ(has_data_out, true);
  auto net_out_node = graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_out_node, nullptr);
  has_data_out = HasOutDataTemp(net_out_node);
  EXPECT_EQ(has_data_out, false);
}

TEST_F(RaggedBinCountFusionPassTest, test_case_02) {
  ge::Graph graph("ragged_bin_count_fusion_pass_test_1");

  auto splits_shape = std::vector<int64_t>({6});
  ge::TensorDesc splits_tensor(ge::Shape(splits_shape), FORMAT_ND, DT_INT64);
  auto splits_data = op::Data("splits");
  splits_data.update_input_desc_x(splits_tensor);
  splits_data.update_output_desc_y(splits_tensor);

  auto values_shape = std::vector<int64_t>({10});
  ge::TensorDesc values_tensor(ge::Shape(values_shape), FORMAT_ND, DT_INT32);
  auto values_data = op::Data("values");
  values_data.update_input_desc_x(values_tensor);
  values_data.update_output_desc_y(values_tensor);

  auto size_shape = std::vector<int64_t>({1});
  ge::TensorDesc size_tensor(ge::Shape(size_shape), FORMAT_ND, DT_INT32);
  auto size_data = op::Data("size");
  size_data.update_input_desc_x(size_tensor);
  size_data.update_output_desc_y(size_tensor);

  auto weights_shape = std::vector<int64_t>({10});
  ge::TensorDesc weights_tensor(ge::Shape(weights_shape), FORMAT_ND, DT_FLOAT);
  auto weights_data = op::Data("weights");
  weights_data.update_input_desc_x(weights_tensor);
  weights_data.update_output_desc_y(weights_tensor);

  auto raggedBinCount = op::RaggedBinCount("RaggedBinCount")
                            .set_input_splits(splits_data)
                            .set_input_values(values_data)
                            .set_input_size(size_data)
                            .set_input_weights(weights_data);
  raggedBinCount.SetAttr("binary_output", true);
  ge::TensorDesc input_desc_splits(ge::Shape({6}), FORMAT_ND, DT_INT64);
  ge::TensorDesc input_desc_values(ge::Shape({10}), FORMAT_ND, DT_INT32);
  ge::TensorDesc input_desc_size(ge::Shape({1}), FORMAT_ND, DT_INT32);
  ge::TensorDesc input_desc_weights(ge::Shape({10}), FORMAT_ND, DT_FLOAT);
  ge::TensorDesc output_desc_output(ge::Shape({5, 5}), FORMAT_ND, DT_FLOAT);
  raggedBinCount.update_input_desc_splits(input_desc_splits);
  raggedBinCount.update_input_desc_values(input_desc_values);
  raggedBinCount.update_input_desc_size(input_desc_size);
  raggedBinCount.update_input_desc_weights(input_desc_weights);
  raggedBinCount.update_output_desc_output(output_desc_output);

  std::vector<Operator> inputs{splits_data, values_data, size_data, weights_data};
  std::vector<Operator> outputs{raggedBinCount};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("RaggedBinCountFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findMinimum = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Minimum") {
      findMinimum = true;
    }
  }

  PlatformInfo platform_info;
  if (platform_info.str_info.short_soc_version != "Ascend910") {
    EXPECT_EQ(findMinimum, false);
  } else {
    EXPECT_EQ(findMinimum, true);
  }
}