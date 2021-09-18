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
#include "nn_norm_ops.h"
#include "reduce_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class softmax_arg_max_value_onnx_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "softmax_arg_max_value_onnx_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "softmax_arg_max_value_onnx_fusion_test TearDown" << std::endl;
  }
};

/**
 * Test case1:
 *            x                           x
 *         /     \                        |
 *   SoftmaxV2  ArgMaxD               SoftmaxV2
 *      |          |      ==>             |
 *  ReduceMaxD     |               ArgMaxWithValue
 *      |       output1               /       \
 *   output0                      output0   output1
 */
TEST_F(softmax_arg_max_value_onnx_fusion_test, softmax_arg_max_value_onnx_fusion_test_01) {
  auto inputData = op::Data("data_as_trans_TransData_350");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 128, 9140}), FORMAT_ND, ge::DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 128, 9140}));
    xDesc.SetOriginFormat(FORMAT_NCHW);

    inputData.update_input_desc_x(xDesc);
    inputData.update_output_desc_y(xDesc);
  }

  auto softmax = op::SoftmaxV2("Softmax_406");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 128, 9140}), FORMAT_NCHW, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 128, 9140}));
    xDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc yDesc(ge::Shape({8, 128, 9140}), FORMAT_NCHW, DT_FLOAT16);
    yDesc.SetOriginShape(ge::Shape({8, 128, 9140}));
    yDesc.SetOriginFormat(FORMAT_NCHW);

    softmax.update_input_desc_x(xDesc);
    softmax.update_output_desc_y(yDesc);
    softmax.set_attr_axes({2});
  }

  auto argMax = op::ArgMaxD("PartitionedCall_ArgMaxV2_57_134");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 128, 9140}), FORMAT_NCHW, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 128, 9140}));
    xDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc yDesc(ge::Shape({8, 128}), FORMAT_NCHW, DT_INT32);
    yDesc.SetOriginShape(ge::Shape({8, 128}));
    yDesc.SetOriginFormat(FORMAT_NCHW);

    argMax.update_input_desc_x(xDesc);
    argMax.update_output_desc_y(yDesc);
    argMax.set_attr_dimension(2);
  }

  auto reduceMax = op::ReduceMaxD("PartitionedCall_ReduceMax_46_95");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 128, 9140}), FORMAT_NCHW, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 128, 9140}));
    xDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc yDesc(ge::Shape({8, 128}), FORMAT_NCHW, DT_FLOAT16);
    yDesc.SetOriginShape(ge::Shape({8, 128}));
    yDesc.SetOriginFormat(FORMAT_NCHW);

    reduceMax.update_input_desc_x(xDesc);
    reduceMax.update_output_desc_y(yDesc);
    reduceMax.set_attr_axes({2});
    reduceMax.set_attr_keep_dims(false);
  }

  auto cast1 = op::Cast("cast_as_NetOutput");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 128}), FORMAT_NCHW, DT_INT32);
    xDesc.SetOriginShape(ge::Shape({8, 128}));
    xDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc yDesc(ge::Shape({8, 128}), FORMAT_NCHW, DT_INT64);
    yDesc.SetOriginShape(ge::Shape({8, 128}));
    yDesc.SetOriginFormat(FORMAT_NCHW);

    cast1.update_input_desc_x(xDesc);
    cast1.update_output_desc_y(yDesc);
  }

  auto cast2 = op::Cast("trans_Cast_352");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 128}), FORMAT_NCHW, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 128}));
    xDesc.SetOriginFormat(FORMAT_NCHW);

    ge::TensorDesc yDesc(ge::Shape({8, 128}), FORMAT_NCHW, DT_FLOAT);
    yDesc.SetOriginShape(ge::Shape({8, 128}));
    yDesc.SetOriginFormat(FORMAT_NCHW);

    cast2.update_input_desc_x(xDesc);
    cast2.update_output_desc_y(yDesc);
  }

  softmax.set_input_x(inputData);
  argMax.set_input_x(inputData);
  reduceMax.set_input_x(softmax);
  cast1.set_input_x(argMax);
  cast2.set_input_x(reduceMax);

  std::vector<Operator> inputs{inputData};
  std::vector<Operator> outputs{cast1, cast2};

  std::string testCaseName = "softmax_arg_max_value_onnx_fusion_test_01";
  ge::Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraph = ge::GraphUtils::GetComputeGraph(graph);

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxArgMaxValueONNXFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findArgMax = false;
  bool findReduceMax = false;
  bool findArgMaxWithValue = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "ArgMaxD") {
      findArgMax = true;
    } else if (node->GetType() == "ReduceMaxD") {
      findReduceMax = true;
    } else if (node->GetType() == "ArgMaxWithValue") {
      findArgMaxWithValue = true;
    }
  }
  EXPECT_EQ(findArgMax, false);
  EXPECT_EQ(findReduceMax, false);
  EXPECT_EQ(findArgMaxWithValue, true);
}

/**
 * Test case2:
 *            x                           x
 *            |                           |
 *        SoftmaxV2                   SoftmaxV2
 *         /     \        ==>             |
 *  ReduceMaxD  ArgMaxD            ArgMaxWithValue
 *      |         |                    /       \
 *   output0    output1            output0   output1
 */
TEST_F(softmax_arg_max_value_onnx_fusion_test, softmax_arg_max_value_onnx_fusion_test_02) {
  auto inputData = op::Data("data_as_trans_TransData_258");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 8}), FORMAT_NHWC, ge::DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 8}));
    xDesc.SetOriginFormat(FORMAT_NHWC);

    inputData.update_input_desc_x(xDesc);
    inputData.update_output_desc_y(xDesc);
  }

  auto softmax = op::SoftmaxV2("Softmax_326");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 8}), FORMAT_NHWC, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 8}));
    xDesc.SetOriginFormat(FORMAT_NHWC);

    ge::TensorDesc yDesc(ge::Shape({8, 8}), FORMAT_NHWC, DT_FLOAT16);
    yDesc.SetOriginShape(ge::Shape({8, 8}));
    yDesc.SetOriginFormat(FORMAT_NHWC);

    softmax.update_input_desc_x(xDesc);
    softmax.update_output_desc_y(yDesc);
    softmax.set_attr_axes({1});
  }

  auto argMax = op::ArgMaxD("PartitionedCall_ArgMaxV2_45_92");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 8}), FORMAT_ND, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 8}));
    xDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc yDesc(ge::Shape({8}), FORMAT_ND, DT_INT32);
    yDesc.SetOriginShape(ge::Shape({8}));
    yDesc.SetOriginFormat(FORMAT_ND);

    argMax.update_input_desc_x(xDesc);
    argMax.update_output_desc_y(yDesc);
    argMax.set_attr_dimension(1);
  }

  auto reduceMax = op::ReduceMaxD("PartitionedCall_ReduceMax_46_95");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 8}), FORMAT_ND, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 8}));
    xDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc yDesc(ge::Shape({8}), FORMAT_ND, DT_FLOAT16);
    yDesc.SetOriginShape(ge::Shape({8}));
    yDesc.SetOriginFormat(FORMAT_ND);

    reduceMax.update_input_desc_x(xDesc);
    reduceMax.update_output_desc_y(yDesc);
    reduceMax.set_attr_axes({1});
    reduceMax.set_attr_keep_dims(false);
  }

  auto cast1 = op::Cast("cast_as_NetOutput");
  {
    ge::TensorDesc xDesc(ge::Shape({8}), FORMAT_ND, DT_INT32);
    xDesc.SetOriginShape(ge::Shape({8}));
    xDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc yDesc(ge::Shape({8}), FORMAT_ND, DT_INT64);
    yDesc.SetOriginShape(ge::Shape({8}));
    yDesc.SetOriginFormat(FORMAT_ND);

    cast1.update_input_desc_x(xDesc);
    cast1.update_output_desc_y(yDesc);
  }

  auto cast2 = op::Cast("trans_Cast_261");
  {
    ge::TensorDesc xDesc(ge::Shape({8}), FORMAT_ND, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8}));
    xDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc yDesc(ge::Shape({8}), FORMAT_ND, DT_FLOAT);
    yDesc.SetOriginShape(ge::Shape({8}));
    yDesc.SetOriginFormat(FORMAT_ND);

    cast2.update_input_desc_x(xDesc);
    cast2.update_output_desc_y(yDesc);
  }

  softmax.set_input_x(inputData);
  argMax.set_input_x(softmax);
  reduceMax.set_input_x(softmax);
  cast1.set_input_x(argMax);
  cast2.set_input_x(reduceMax);

  std::vector<Operator> inputs{inputData};
  std::vector<Operator> outputs{cast1, cast2};

  std::string testCaseName = "softmax_arg_max_value_onnx_fusion_test_02";
  ge::Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraph = ge::GraphUtils::GetComputeGraph(graph);

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxArgMaxValueONNXFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findArgMax = false;
  bool findReduceMax = false;
  bool findArgMaxWithValue = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "ArgMaxD") {
      findArgMax = true;
    } else if (node->GetType() == "ReduceMaxD") {
      findReduceMax = true;
    } else if (node->GetType() == "ArgMaxWithValue") {
      findArgMaxWithValue = true;
    }
  }
  EXPECT_EQ(findArgMax, false);
  EXPECT_EQ(findReduceMax, false);
  EXPECT_EQ(findArgMaxWithValue, true);
}

/**
 * Test without ArgMaxD.
 */
TEST_F(softmax_arg_max_value_onnx_fusion_test, softmax_arg_max_value_onnx_fusion_test_03) {
  auto inputData = op::Data("data_as_trans_TransData_258");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 8}), FORMAT_NHWC, ge::DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 8}));
    xDesc.SetOriginFormat(FORMAT_NHWC);

    inputData.update_input_desc_x(xDesc);
    inputData.update_output_desc_y(xDesc);
  }

  auto softmax = op::SoftmaxV2("Softmax_326");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 8}), FORMAT_NHWC, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 8}));
    xDesc.SetOriginFormat(FORMAT_NHWC);

    ge::TensorDesc yDesc(ge::Shape({8, 8}), FORMAT_NHWC, DT_FLOAT16);
    yDesc.SetOriginShape(ge::Shape({8, 8}));
    yDesc.SetOriginFormat(FORMAT_NHWC);

    softmax.update_input_desc_x(xDesc);
    softmax.update_output_desc_y(yDesc);
    softmax.set_attr_axes({1});
  }

  auto reduceMax = op::ReduceMaxD("PartitionedCall_ReduceMax_46_95");
  {
    ge::TensorDesc xDesc(ge::Shape({8, 8}), FORMAT_ND, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8, 8}));
    xDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc yDesc(ge::Shape({8}), FORMAT_ND, DT_FLOAT16);
    yDesc.SetOriginShape(ge::Shape({8}));
    yDesc.SetOriginFormat(FORMAT_ND);

    reduceMax.update_input_desc_x(xDesc);
    reduceMax.update_output_desc_y(yDesc);
    reduceMax.set_attr_axes({1});
    reduceMax.set_attr_keep_dims(false);
  }

  auto cast2 = op::Cast("trans_Cast_261");
  {
    ge::TensorDesc xDesc(ge::Shape({8}), FORMAT_ND, DT_FLOAT16);
    xDesc.SetOriginShape(ge::Shape({8}));
    xDesc.SetOriginFormat(FORMAT_ND);

    ge::TensorDesc yDesc(ge::Shape({8}), FORMAT_ND, DT_FLOAT);
    yDesc.SetOriginShape(ge::Shape({8}));
    yDesc.SetOriginFormat(FORMAT_ND);

    cast2.update_input_desc_x(xDesc);
    cast2.update_output_desc_y(yDesc);
  }

  softmax.set_input_x(inputData);
  reduceMax.set_input_x(softmax);
  cast2.set_input_x(reduceMax);

  std::vector<Operator> inputs{inputData};
  std::vector<Operator> outputs{cast2};

  std::string testCaseName = "softmax_arg_max_value_onnx_fusion_test_03";
  ge::Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr computeGraph = ge::GraphUtils::GetComputeGraph(graph);

  GE_DUMP(computeGraph, testCaseName + "_before_fusion");
  fe::FusionPassTestUtils::RunGraphFusionPass("SoftmaxArgMaxValueONNXFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findArgMax = false;
  bool findReduceMax = false;
  bool findArgMaxWithValue = false;
  for (auto node : computeGraph->GetAllNodes()) {
    if (node->GetType() == "ArgMaxD") {
      findArgMax = true;
    } else if (node->GetType() == "ReduceMaxD") {
      findReduceMax = true;
    } else if (node->GetType() == "ArgMaxWithValue") {
      findArgMaxWithValue = true;
    }
  }
  EXPECT_EQ(findArgMax, false);
  EXPECT_EQ(findReduceMax, true);
  EXPECT_EQ(findArgMaxWithValue, false);
}
