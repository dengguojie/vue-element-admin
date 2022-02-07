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

using namespace ge;
using namespace op;

class concatv2d_sliced_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "concatv2d_sliced_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "concatv2d_sliced_fusion_test TearDown" << std::endl;
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

TEST_F(concatv2d_sliced_fusion_test, concatv2d_sliced_fusion_test_01) {
  std::string testCaseName = "concatv2d_sliced_fusion_test_01";

  auto x0 = CreateDataNode("x0", {1, 6553600}, FORMAT_ND, DT_FLOAT);
  auto x1 = CreateDataNode("x1", {1, 13107200}, FORMAT_ND, DT_FLOAT);

  auto concatOp = ConcatV2D("ConcatV2D_11");
  concatOp.create_dynamic_input_x(2)
      .set_dynamic_input_x(0, x0)
      .set_dynamic_input_x(1, x1)
      .set_attr_concat_dim(1)
      .set_attr_N(2);
  concatOp.update_output_desc_y(SimpleTensorDesc("y", {1, 19660800}, FORMAT_ND, DT_FLOAT));

  auto slice1 = SliceD("SliceD_11");
  slice1.set_input_x(concatOp)
      .set_attr_offsets({0, 0})
      .set_attr_size({1, 6553600});

  auto slice2 = SliceD("SliceD_12");
  slice2.set_input_x(concatOp)
      .set_attr_offsets({0, 6553600})
      .set_attr_size({1, 13107200});

  auto endOp = Maximum("Maximum_11");
  endOp.set_input_x1(slice1).set_input_x2(slice2);

  std::vector<Operator> inputs{x0, x1};
  std::vector<Operator> outputs{endOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ZSliceDConcatV2DFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findConcatV2DNode = false;
  bool findSliceDNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "ConcatV2D") {
      findConcatV2DNode = true;
    } else if (iNode->GetType() == "SliceD") {
      findSliceDNode = true;
    }
  }
  EXPECT_EQ(findConcatV2DNode, false);
  EXPECT_EQ(findSliceDNode, false);
}

TEST_F(concatv2d_sliced_fusion_test, concatv2d_sliced_fusion_test_02) {
  std::string testCaseName = "concatv2d_sliced_fusion_test_02";

  auto x0 = CreateDataNode("x0", {1, 6553600}, FORMAT_ND, DT_FLOAT);
  auto x1 = CreateDataNode("x1", {1, 13107200}, FORMAT_ND, DT_FLOAT);

  auto concatOp = ConcatV2D("ConcatV2D_21");
  concatOp.create_dynamic_input_x(2)
      .set_dynamic_input_x(0, x0)
      .set_dynamic_input_x(1, x1)
      .set_attr_concat_dim(1)
      .set_attr_N(2);
  concatOp.update_output_desc_y(SimpleTensorDesc("y", {1, 19660800}, FORMAT_ND, DT_FLOAT));

  auto slice1 = SliceD("SliceD_21");
  slice1.set_input_x(concatOp)
      .set_attr_offsets({0, 0})
      .set_attr_size({1, 6553600});

  auto slice2 = SliceD("SliceD_22");
  slice2.set_input_x(concatOp)
      .set_attr_offsets({0, 6553600})
      .set_attr_size({1, 13107200});

  auto endOp = Maximum("Maximum_21");
  endOp.set_input_x1(slice2)
      .set_input_x2(slice1);

  std::vector<Operator> inputs{x0, x1};
  std::vector<Operator> outputs{endOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ZSliceDConcatV2DFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findConcatV2DNode = false;
  bool findSliceDNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "ConcatV2D") {
      findConcatV2DNode = true;
    } else if (iNode->GetType() == "SliceD") {
      findSliceDNode = true;
    }
  }
  EXPECT_EQ(findConcatV2DNode, false);
  EXPECT_EQ(findSliceDNode, false);
}
