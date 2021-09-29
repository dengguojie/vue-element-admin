/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nn_calculation_ops.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"

using namespace ge;

namespace fe {

class a_split_node_remove_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "a_split_node_remove_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "a_split_node_remove_test TearDown" << std::endl;
  }

  /************************************
   *
   *               split
   *       filter    /\     filter
   *           \  /       \ /
   *         conv2d ... conv2d
   *              \       /
   *                 \/
   *              concat(v2)
   *                 |
   *
   *************************************/
  void BuildGraphCascade(ComputeGraphPtr &compute_graph) {
    ge::Graph graph("test");
    auto input0 = op::Data("input0");
    ge::Shape shape_x({1, 7, 7, 16});
    ge::TensorDesc tensor_desc_x(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensor_desc_x);
    input0.update_output_desc_y(tensor_desc_x);

    auto split_dim = op::Const("split_dim");
    Tensor axis;
    int32_t * data_value = new int32_t[1];
    * data_value = 3;
    axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    axis.SetData((uint8_t*)data_value, 4);
    split_dim.set_attr_value(axis);
    split_dim.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto split_layer = op::Split("split");
    split_layer.set_input_split_dim(split_dim)
               .set_input_x(input0)
               .set_attr_num_split(1)
               .create_dynamic_output_y(1);

    TensorDesc filter_desc(ge::Shape({3, 3, 16, 16}), FORMAT_HWCN, DT_FLOAT16);
    filter_desc.SetOriginFormat(FORMAT_HWCN);

    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    fp16_t * filter_value = new fp16_t[3*3*16*16];
    filter.SetTensorDesc(filter_desc);
    filter.SetData((uint8_t*)filter_value, 2*3*3*16*16);
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filter_desc);

    auto conv2d_0_layer = op::Conv2D("conv2d_0");
    conv2d_0_layer.set_input_x(split_layer, "y0")
                  .set_input_filter(filter_0)
                  .set_attr_strides({0, 1, 1, 0})
                  .set_attr_pads({0, 0, 0, 0});

    auto filter_1 = op::Const("filter_1");
    filter_1.set_attr_value(filter);
    filter_1.update_output_desc_y(filter_desc);

    auto conv2d_1_layer = op::Conv2D("conv2d_1");
    conv2d_1_layer.set_input_x(split_layer, "y0")
                  .set_input_filter(filter_1)
                  .set_attr_strides({0, 1, 1, 0})
                  .set_attr_pads({0, 0, 0, 0});

    auto concat_dim = op::Const("concat_dim");
    concat_dim.set_attr_value(axis);
    concat_dim.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto concat_layer = op::Concat("concat");
    concat_layer.set_input_concat_dim(concat_dim)
                .create_dynamic_input_x(2)
                .set_dynamic_input_x(0, conv2d_0_layer)
                .set_dynamic_input_x(1, conv2d_1_layer)
                .set_attr_N(2);

    delete[] data_value;
    delete[] filter_value;

    std::vector<Operator> inputs{input0, split_dim, filter_0, filter_1, concat_dim};
    std::vector<Operator> outputs{concat_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }

  void BuildGraphSingle(ComputeGraphPtr &compute_graph, const vector<int64_t>& shape,
                        int32_t split_dim, int32_t split_num) {
    ge::Graph graph("test_split");
    auto input0 = op::Data("input0");
    ge::Shape shape_x(shape);
    ge::TensorDesc tensor_desc_x(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensor_desc_x);
    input0.update_output_desc_y(tensor_desc_x);

    auto split_dim_op = op::Const("split_dim");
    Tensor axis;
    axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    axis.SetData((uint8_t*)&split_dim, sizeof(split_dim));
    split_dim_op.set_attr_value(axis);
    split_dim_op.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto split_layer = op::Split("split");
    split_layer.set_input_split_dim(split_dim_op)
               .set_input_x(input0)
               .set_attr_num_split(split_num)
               .create_dynamic_output_y(split_num);

    std::vector<Operator> inputs{input0, split_dim_op};
    std::vector<Operator> outputs{split_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(a_split_node_remove_test, a_split_node_remove_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraphCascade(compute_graph);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ASplitNodeRemoveFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool find_split = false;
  bool find_split_d = false;
  bool find_splitv_d = false;
  bool find_concat = false;
  bool find_conv2d = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "Split") {
        find_split = true;
    }
    if (node->GetType() == "SplitD") {
        find_split_d = true;
    }
    if (node->GetType() == "SplitVD") {
        find_splitv_d = true;
    }
    if (node->GetType() == "Concat") {
        find_concat = true;
    }
    if (node->GetType() == "Conv2D") {
        find_conv2d = true;
    }
  }
  EXPECT_EQ(find_split, false);
  EXPECT_EQ(find_split_d, false);
  EXPECT_EQ(find_splitv_d, false);
  EXPECT_EQ(find_concat, true);
  EXPECT_EQ(find_conv2d, true);
}

TEST_F(a_split_node_remove_test, a_split_node_remove_test_2) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraphSingle(compute_graph, {-1, -1, -1, -1}, -1, 1);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ASplitNodeRemoveFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool find_split = false;
  bool find_split_d = false;
  bool find_splitv_d = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "Split") {
      find_split = true;
      continue;
    }
    if (node->GetType() == "SplitD") {
        find_split_d = true;
    }
    if (node->GetType() == "SplitVD") {
        find_splitv_d = true;
    }
  }
  EXPECT_EQ(find_split, false);
  EXPECT_EQ(find_split_d, false);
  EXPECT_EQ(find_splitv_d, false);
}

TEST_F(a_split_node_remove_test, a_split_node_remove_test_3) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraphSingle(compute_graph, {7, 2, 12, 17}, 0, 1);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ASplitNodeRemoveFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool find_split = false;
  bool find_split_d = false;
  bool find_splitv_d = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "Split") {
      find_split = true;
      continue;
    }
    if (node->GetType() == "SplitD") {
        find_split_d = true;
    }
    if (node->GetType() == "SplitVD") {
        find_splitv_d = true;
    }
  }
  EXPECT_EQ(find_split, false);
  EXPECT_EQ(find_split_d, false);
  EXPECT_EQ(find_splitv_d, false);
}
} // namespace fe

