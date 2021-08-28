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
#include "vector_search.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class scanpqcodes_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "scanpqcodes_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "scanpqcodes_fusion_test TearDown" << std::endl;
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

TEST_F(scanpqcodes_fusion_test, scanpqcodes_fusion_test_01) {
  std::string testCaseName = "scanpqcodes_fusion_test_01";

  std::vector<int64_t> dims0({500000, 16});
  auto data0 = CreateDataNode("ivf_4234", dims0, FORMAT_ND, DT_UINT8);

  std::vector<int64_t> dims1({64});
  auto data1 = CreateDataNode("bucket_list_1811", dims1, FORMAT_ND, DT_INT32);

  std::vector<int64_t> dims2({64});
  auto data2 = CreateDataNode("bucket_base_distance_4433", dims2, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> dims3({500000});
  auto data3 = CreateDataNode("bucket_limit_24255", dims3, FORMAT_ND, DT_INT32);

  std::vector<int64_t> dims4({500000});
  auto data4 = CreateDataNode("bucket_offsets_77776", dims4, FORMAT_ND, DT_INT32);

  std::vector<int64_t> dims5({10, 16, 256});
  auto data5 = CreateDataNode("adc_tables_354355", dims5, FORMAT_ND, DT_FLOAT16);

  std::string scanPQCodesOpName = "ScanPQCodes_0";
  auto scanPQCodesOp = ScanPQCodes(scanPQCodesOpName.c_str());
  scanPQCodesOp.set_input_ivf(data0)
      .set_input_bucket_list(data1)
      .set_input_bucket_base_distance(data2)
      .set_input_bucket_limits(data3)
      .set_input_bucket_offsets(data4)
      .set_input_adc_tables(data5)
      .set_attr_total_limit(666)
      .set_attr_group_size(64)
      .set_attr_extreme_mode(0)
      .set_attr_split_count(1)
      .set_attr_split_index(0);
  scanPQCodesOp.update_output_desc_actual_count(SimpleTensorDesc("actual_count_1v1x1", {1024}, FORMAT_ND, DT_INT32));
  scanPQCodesOp.update_output_desc_pq_distance(SimpleTensorDesc("pq_distance_2b4b54f6", {2048}, FORMAT_ND, DT_FLOAT16));
  scanPQCodesOp.update_output_desc_grouped_extreme_distance(
      SimpleTensorDesc("grouped_extreme_distance_8d1j", {1000}, FORMAT_ND, DT_FLOAT16));
  scanPQCodesOp.update_output_desc_pq_ivf(SimpleTensorDesc("pq_ivf_4v6jg", {1000}, FORMAT_ND, DT_INT32));
  scanPQCodesOp.update_output_desc_pq_index(SimpleTensorDesc("pq_index_9x6s8a", {2500}, FORMAT_ND, DT_INT32));

  auto topkOp = TopKPQDistance("TopKPQDistance_0");

  topkOp.create_dynamic_input_actual_count(1);
  topkOp.set_dynamic_input_actual_count(0, scanPQCodesOp, "actual_count");

  topkOp.create_dynamic_input_pq_distance(1);
  topkOp.set_dynamic_input_pq_distance(0, scanPQCodesOp, "pq_distance");

  topkOp.create_dynamic_input_grouped_extreme_distance(1);
  topkOp.set_dynamic_input_grouped_extreme_distance(0, scanPQCodesOp, "grouped_extreme_distance");

  topkOp.create_dynamic_input_pq_ivf(1);
  topkOp.set_dynamic_input_pq_ivf(0, scanPQCodesOp, "pq_ivf");

  topkOp.create_dynamic_input_pq_index(1);
  topkOp.set_dynamic_input_pq_index(0, scanPQCodesOp, "pq_index");

  topkOp.update_output_desc_topk_distance(SimpleTensorDesc("topk_distance_7x7s7s7", {2100}, FORMAT_ND, DT_FLOAT16));
  topkOp.update_output_desc_topk_ivf(SimpleTensorDesc("topk_ivf_8asd3fa3", {2200}, FORMAT_ND, DT_INT32));
  topkOp.update_output_desc_topk_index(SimpleTensorDesc("topk_index_45x4a9", {2300}, FORMAT_ND, DT_INT32));

  topkOp.set_attr_order("ASC").set_attr_k(0).set_attr_group_size(0);

  auto endOp0 = ConcatV2D("end_op_0");
  endOp0.create_dynamic_input_x(1);
  endOp0.set_dynamic_input_x(0, topkOp, "topk_distance");
  endOp0.set_attr_concat_dim(0);

  auto endOp1 = ConcatV2D("end_op_1");
  endOp1.create_dynamic_input_x(2);
  endOp1.set_dynamic_input_x(0, topkOp, "topk_ivf");
  endOp1.set_dynamic_input_x(1, topkOp, "topk_index");
  endOp1.set_attr_concat_dim(0);

  std::vector<Operator> inputs{data0, data1, data2, data3, data4, data5};
  std::vector<Operator> outputs{endOp0, endOp1};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ScanPQCodesFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findAiCoreNode = false;
  bool findVectorCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "ScanPQCodes") {
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

TEST_F(scanpqcodes_fusion_test, scanpqcodes_fusion_test_02) {
  std::string testCaseName = "scanpqcodes_fusion_test_02";

  std::vector<int64_t> dims0({-1, 16});
  auto data0 = CreateDataNode("ivf_4234", dims0, FORMAT_ND, DT_UINT8);

  std::vector<int64_t> dims1({64});
  auto data1 = CreateDataNode("bucket_list_1811", dims1, FORMAT_ND, DT_INT32);

  std::vector<int64_t> dims2({-1});
  auto data2 = CreateDataNode("bucket_base_distance_4433", dims2, FORMAT_ND, DT_FLOAT16);

  std::vector<int64_t> dims3({500000});
  auto data3 = CreateDataNode("bucket_limit_24255", dims3, FORMAT_ND, DT_INT32);

  std::vector<int64_t> dims4({-1});
  auto data4 = CreateDataNode("bucket_offsets_77776", dims4, FORMAT_ND, DT_INT32);

  std::vector<int64_t> dims5({-1, 16, 256});
  auto data5 = CreateDataNode("adc_tables_354355", dims5, FORMAT_ND, DT_FLOAT16);

  std::string scanPQCodesOpName = "ScanPQCodes_0";
  auto scanPQCodesOp = ScanPQCodes(scanPQCodesOpName.c_str());
  scanPQCodesOp.set_input_ivf(data0)
      .set_input_bucket_list(data1)
      .set_input_bucket_base_distance(data2)
      .set_input_bucket_limits(data3)
      .set_input_bucket_offsets(data4)
      .set_input_adc_tables(data5)
      .set_attr_total_limit(2237)
      .set_attr_group_size(64)
      .set_attr_extreme_mode(0)
      .set_attr_split_count(1)
      .set_attr_split_index(0);
  scanPQCodesOp.update_output_desc_actual_count(SimpleTensorDesc("actual_count_1v1x1", {1024}, FORMAT_ND, DT_INT32));
  scanPQCodesOp.update_output_desc_pq_distance(SimpleTensorDesc("pq_distance_2b4b54f6", {2048}, FORMAT_ND, DT_FLOAT16));
  scanPQCodesOp.update_output_desc_grouped_extreme_distance(
      SimpleTensorDesc("grouped_extreme_distance_8d1j", {1000}, FORMAT_ND, DT_FLOAT16));
  scanPQCodesOp.update_output_desc_pq_ivf(SimpleTensorDesc("pq_ivf_4v6jg", {1000}, FORMAT_ND, DT_INT32));
  scanPQCodesOp.update_output_desc_pq_index(SimpleTensorDesc("pq_index_9x6s8a", {2500}, FORMAT_ND, DT_INT32));

  auto topkOp = TopKPQDistance("TopKPQDistance_0");

  topkOp.create_dynamic_input_actual_count(1);
  topkOp.set_dynamic_input_actual_count(0, scanPQCodesOp, "actual_count");

  topkOp.create_dynamic_input_pq_distance(1);
  topkOp.set_dynamic_input_pq_distance(0, scanPQCodesOp, "pq_distance");

  topkOp.create_dynamic_input_grouped_extreme_distance(1);
  topkOp.set_dynamic_input_grouped_extreme_distance(0, scanPQCodesOp, "grouped_extreme_distance");

  topkOp.create_dynamic_input_pq_ivf(1);
  topkOp.set_dynamic_input_pq_ivf(0, scanPQCodesOp, "pq_ivf");

  topkOp.create_dynamic_input_pq_index(1);
  topkOp.set_dynamic_input_pq_index(0, scanPQCodesOp, "pq_index");

  topkOp.update_output_desc_topk_distance(SimpleTensorDesc("topk_distance_7x7s7s7", {2100}, FORMAT_ND, DT_FLOAT16));
  topkOp.update_output_desc_topk_ivf(SimpleTensorDesc("topk_ivf_8asd3fa3", {2200}, FORMAT_ND, DT_INT32));
  topkOp.update_output_desc_topk_index(SimpleTensorDesc("topk_index_45x4a9", {2300}, FORMAT_ND, DT_INT32));

  topkOp.set_attr_order("ASC").set_attr_k(0).set_attr_group_size(0);

  auto endOp0 = Adds("end_op_0");
  endOp0.set_input_x_by_name(topkOp, "topk_distance");
  endOp0.set_attr_value(100.0);

  auto endOp1 = Add("end_op_1");
  endOp1.set_input_x1_by_name(topkOp, "topk_ivf");
  endOp1.set_input_x2_by_name(topkOp, "topk_index");

  std::vector<Operator> inputs{data0, data1, data2, data3, data4, data5};
  std::vector<Operator> outputs{endOp0, endOp1};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ScanPQCodesFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  bool findAiCoreNode = false;
  bool findVectorCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "ScanPQCodes") {
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
