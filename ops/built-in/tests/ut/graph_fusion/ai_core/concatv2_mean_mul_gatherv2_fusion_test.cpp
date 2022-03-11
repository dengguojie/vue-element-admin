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
#include "reduce_ops.h"
#include "split_combination_ops.h"
#include "fusion_pass_test_utils.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class concatv2_mean_mul_gatherv2_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "concatv2_mean_mul_gatherv2_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "concatv2_mean_mul_gatherv2_fusion_test TearDown" << std::endl;
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

TEST_F(concatv2_mean_mul_gatherv2_fusion_test, concatv2_mean_mul_gatherv2_fusion_test_01) {
  std::cout << "concatv2_mean_mul_gatherv2_fusion_test_01  start" << std::endl;
  std::string testCaseName = "concatv2_mean_mul_gatherv2_fusion_test_01";

  // set soc_version
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 8;
  platformInfo.soc_info.vector_core_cnt = 7;
  optiCompilationInfo.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  // ExpandDims
  auto expandDims_01_x = CreateDataNode("expandDims_01_x", {640, 1}, FORMAT_ND, DT_FLOAT);
  auto expandDims_02_x = CreateDataNode("expandDims_02_x", {640, 1}, FORMAT_ND, DT_FLOAT);

  ge::Tensor axis_tensor;
  std::vector<int64_t> axis_vec{1};
  ge::Shape axis_shape(axis_vec);
  ge::TensorDesc axis_desc(axis_shape, FORMAT_ND, DT_INT32);
  int64_t axis_size = axis_desc.GetShape().GetShapeSize();
  axis_desc.SetSize(axis_size * sizeof(int32_t));
  axis_tensor.SetTensorDesc(axis_desc);

  int32_t* axis_data = nullptr;
  axis_data = new int32_t[axis_size];
  *(axis_data + 0) = 2;
  axis_tensor.SetData((uint8_t*)axis_data, axis_size * sizeof(int32_t));
  delete[] axis_data;
  auto axis = op::Constant("axis").set_attr_value(axis_tensor);
  axis.update_output_desc_y(axis_desc);

  auto expandDims_01 = ExpandDims("ExpandDims_01");
  expandDims_01.set_input_x(expandDims_01_x).set_input_axis(axis);
  expandDims_01.update_output_desc_y(SimpleTensorDesc("expandDims_01_y", {640, 1, 1}, FORMAT_ND, DT_FLOAT));
  auto expandDims_02 = ExpandDims("expandDims_02");
  expandDims_02.set_input_x(expandDims_02_x).set_input_axis(axis);
  expandDims_02.update_output_desc_y(SimpleTensorDesc("expandDims_02_y", {640, 1, 1}, FORMAT_ND, DT_FLOAT));

  // GatherV2
  int32_t* gatherv2_axis_data = nullptr;
  gatherv2_axis_data = new int32_t[axis_size];
  *(gatherv2_axis_data + 0) = 0;
  axis_tensor.SetData((uint8_t*)gatherv2_axis_data, axis_size * sizeof(int32_t));
  delete[] gatherv2_axis_data;
  auto gatherv2_axis = op::Constant("gatherv2_axis").set_attr_value(axis_tensor);
  gatherv2_axis.update_output_desc_y(axis_desc);

  auto gatherv2_01_data = CreateDataNode("gatherv2_01_data", {10, 110}, FORMAT_ND, DT_FLOAT);
  auto gatherv2_01_indices = CreateDataNode("gatherv2_01_indices", {640, 1}, FORMAT_ND, DT_INT64);
  auto gatherV2_01 = op::GatherV2("GatherV2_01");
  gatherV2_01.set_input_x(gatherv2_01_data).set_input_indices(gatherv2_01_indices).set_input_axis(gatherv2_axis);
  gatherV2_01.update_output_desc_y(SimpleTensorDesc("gatherV2_01_y", {640, 1, 110}, FORMAT_ND, DT_FLOAT));

  auto gatherv2_02_data = CreateDataNode("gatherv2_02_data", {10, 110}, FORMAT_ND, DT_FLOAT);
  auto gatherv2_02_indices = CreateDataNode("gatherv2_02_indices", {640, 1}, FORMAT_ND, DT_INT64);
  auto gatherV2_02 = op::GatherV2("GatherV2_02");
  gatherV2_02.set_input_x(gatherv2_02_data).set_input_indices(gatherv2_02_indices).set_input_axis(gatherv2_axis);
  gatherV2_02.update_output_desc_y(SimpleTensorDesc("gatherV2_02_y", {640, 1, 110}, FORMAT_ND, DT_FLOAT));

  // Mul
  auto mul_01 = op::Mul("mul_01");
  mul_01.set_input_x1(expandDims_01).set_input_x2(gatherV2_01);
  mul_01.update_output_desc_y(SimpleTensorDesc("mul_01_y", {640, 1, 110}, FORMAT_ND, DT_FLOAT));

  auto mul_02 = op::Mul("mul_02");
  mul_02.set_input_x1(expandDims_02).set_input_x2(gatherV2_02);
  mul_02.update_output_desc_y(SimpleTensorDesc("mul_02_y", {640, 1, 110}, FORMAT_ND, DT_FLOAT));
  std::cout << "concatv2_mean_mul_gatherv2_fusion_test_01  mul_01, mul_02 end" << std::endl;

  // ReduceMean
  int32_t* reduceMean_axes_data = nullptr;
  reduceMean_axes_data = new int32_t[axis_size];
  *(reduceMean_axes_data + 0) = 1;
  axis_tensor.SetData((uint8_t*)reduceMean_axes_data, axis_size * sizeof(int32_t));
  delete[] reduceMean_axes_data;
  auto reduceMean_axes = op::Constant("reduceMean_axes").set_attr_value(axis_tensor);
  reduceMean_axes.update_output_desc_y(axis_desc);

  auto reduceMean_01 = op::ReduceMean("reduceMean_01");
  reduceMean_01.set_input_x(mul_01).set_input_axes(reduceMean_axes).set_attr_keep_dims(false);
  reduceMean_01.update_output_desc_y(SimpleTensorDesc("reduceMean_01_y", {640, 110}, FORMAT_ND, DT_FLOAT));

  auto reduceMean_02 = op::ReduceMean("reduceMean_02");
  reduceMean_02.set_input_x(mul_02).set_input_axes(reduceMean_axes).set_attr_keep_dims(false);
  reduceMean_02.update_output_desc_y(SimpleTensorDesc("reduceMean_02_y", {640, 110}, FORMAT_ND, DT_FLOAT));

  // ConcatV2D
  auto concatV2d_01 = op::ConcatV2D("concatv2d_01");
  concatV2d_01.create_dynamic_input_x(2)
      .set_dynamic_input_x(0, reduceMean_01)
      .set_dynamic_input_x(1, reduceMean_02)
      .set_attr_concat_dim(1)
      .set_attr_N(2);
  concatV2d_01.update_output_desc_y(SimpleTensorDesc("concatV2d_01_y", {640, 220}, FORMAT_ND, DT_FLOAT));

  // graph
  std::vector<Operator> inputs{expandDims_01_x, gatherv2_01_data, gatherv2_01_indices,
                               expandDims_02_x, gatherv2_02_data, gatherv2_02_indices,};
  std::vector<Operator> outputs{concatV2d_01};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("AConcatV2MeanMulGatherV2FusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  std::vector<std::string> engineNameVec;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "ReduceMean") {
      ge::OpDescPtr iOpDesc = iNode->GetOpDesc();
      std::string specifiedEngineName;
      ge::AttrUtils::GetStr(iOpDesc, "_specified_engine_name", specifiedEngineName);
      engineNameVec.push_back(specifiedEngineName);
    }
  }

  ASSERT_TRUE((engineNameVec[0] == "VectorEngine") || (engineNameVec[1] == "VectorEngine"));
  std::cout << "concatv2_mean_mul_gatherv2_fusion_test_01  complete" << std::endl;
}
