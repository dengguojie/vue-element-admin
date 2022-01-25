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
#include "deep_md.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class tabulatefusion_transpose_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tabulatefusion_transpose_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tabulatefusion_transpose_fusion_test TearDown" << std::endl;
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

TEST_F(tabulatefusion_transpose_fusion_test, tabulatefusion_transpose_fusion_test_01) {
  std::string testCaseName = "tabulatefusion_transpose_fusion_test_01";
  int64_t nloc = 8192;
  int64_t nnei = 92;
  int32_t last_layer_size = 100;
  int64_t table_dim0 = 2;

  ge::Tensor table_tensor;
  std::vector<int64_t> dims_table{table_dim0, last_layer_size * 6};
  ge::Shape shape_table(dims_table);
  ge::TensorDesc tensorDescTable(shape_table, FORMAT_ND, DT_FLOAT);
  int64_t table_size = tensorDescTable.GetShape().GetShapeSize();
  tensorDescTable.SetSize(table_size * sizeof(float));
  table_tensor.SetTensorDesc(tensorDescTable);
  float* table_data = nullptr;
  table_data = new float[table_size];
  table_tensor.SetData((uint8_t*)table_data, table_size * sizeof(float));
  delete [] table_data;
  auto table = op::Constant("table");
  table.set_attr_value(table_tensor);
  table.update_output_desc_y(tensorDescTable);

  auto table_info = CreateDataNode("table_info", {6}, FORMAT_ND, DT_FLOAT);
  auto em_x = CreateDataNode("em_x", {nloc, nnei}, FORMAT_ND, DT_FLOAT);
  auto em = CreateDataNode("em", {nloc, nnei, 4}, FORMAT_ND, DT_FLOAT);

  std::string tabulatefusionOpName = "TabulateFusion_01";
  auto tabulatefusionOp = TabulateFusion(tabulatefusionOpName.c_str());
  tabulatefusionOp.set_input_table(table)
      .set_input_table_info(table_info)
      .set_input_em_x(em_x)
      .set_input_em(em)
      .set_attr_last_layer_size(last_layer_size);
  tabulatefusionOp.update_output_desc_descriptor(SimpleTensorDesc("descriptor", {nloc, 4, last_layer_size},
                                                 FORMAT_ND, DT_FLOAT));

  std::vector<Operator> inputs{table_info, em_x, em};
  std::vector<Operator> outputs{tabulatefusionOp};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ATabulateFusionTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *computeGraph);

  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "TabulateFusion") {
      ge::OpDescPtr iOpDesc = iNode->GetOpDesc();

      ge::GeTensorDesc iutputTable = iOpDesc->GetInputDesc(0);
      ge::GeShape iutputTableShape = iutputTable.GetShape();
      EXPECT_EQ(iutputTableShape.GetDim(1), 128 * 6);
    }
  }
}
