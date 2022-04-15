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
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "nn_norm_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class attention_score_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "attention_score_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "attention_score_fusion_pass_test TearDown" << std::endl;
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

TEST_F(attention_score_fusion_pass_test, attention_score_fusion_pass_test_01) {
  std::string testCaseName = "attention_score_fusion_pass_test_01";
  int32_t batch_dim1 = 32;
  int32_t batch_dim2 = 16;
  int32_t seq_num = 24;
  int32_t nz_dim = 16;
  int32_t n_num = 4;

  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 1;
  platformInfo.str_info.ccec_aic_version = "dav-s200";
  optiCompilationInfo.soc_version = "Ascend710";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

  auto query = CreateDataNode("query", {batch_dim1, batch_dim2, seq_num * nz_dim, n_num * nz_dim}, FORMAT_ND, DT_FLOAT16);
  auto key = CreateDataNode("key", {batch_dim1, batch_dim2, seq_num * nz_dim, n_num * nz_dim}, FORMAT_ND, DT_FLOAT16);
  auto value = CreateDataNode("value", {batch_dim1, batch_dim2, seq_num * nz_dim, n_num * nz_dim}, FORMAT_ND, DT_FLOAT16);
  auto pad_mask = CreateDataNode("padding_mask", {batch_dim1, 1, seq_num * nz_dim, seq_num * nz_dim}, FORMAT_ND, DT_FLOAT16);
  auto scale = CreateDataNode("scale", {1}, FORMAT_ND, DT_FLOAT16);
  auto drop_mask = CreateDataNode("drop_mask", {batch_dim1, 1, seq_num * nz_dim, seq_num * nz_dim}, FORMAT_ND, DT_FLOAT16);

  ge::Shape shape_x({batch_dim1, batch_dim2, seq_num * nz_dim, seq_num * nz_dim});
  ge::Shape shape_y({batch_dim1, batch_dim2, seq_num * nz_dim, n_num * nz_dim});
  ge::Shape shape_con({12288, 1024});
  ge::TensorDesc tensor_desc_bmm1_output(shape_x, FORMAT_ND,  DT_FLOAT16);
  tensor_desc_bmm1_output.SetOriginShape(shape_x);
  tensor_desc_bmm1_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_fused_mul_add_output(shape_x, FORMAT_ND,  DT_FLOAT16);
  tensor_desc_fused_mul_add_output.SetOriginShape(shape_x);
  tensor_desc_fused_mul_add_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_softmax_output(shape_x, FORMAT_ND,  DT_FLOAT16);
  tensor_desc_softmax_output.SetOriginShape(shape_x);
  tensor_desc_softmax_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_bmm2_output(shape_y, FORMAT_ND,  DT_FLOAT16);
  tensor_bmm2_output.SetOriginShape(shape_y);
  tensor_bmm2_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_confusion_output(shape_con, FORMAT_ND,  DT_FLOAT16);
  tensor_confusion_output.SetOriginShape(shape_con);
  tensor_confusion_output.SetOriginFormat(FORMAT_ND);

  auto bmm1_op = op::BatchMatMulV2("batchmatmul_v2_01");
  bmm1_op.set_input_x1(query)
         .set_input_x2(key)
         .set_attr_adj_x1(false)
         .set_attr_adj_x2(true);
  bmm1_op.update_output_desc_y(tensor_desc_bmm1_output);

  auto mul_op = op::FusedMulAdd("fused_mul_add");
  mul_op.set_input_x1(bmm1_op)
        .set_input_x2(scale)
        .set_input_x3(pad_mask);
  mul_op.update_output_desc_y(tensor_desc_fused_mul_add_output);

  auto softmax_op = op::SoftmaxV2("softmax_v2_01");
  softmax_op.set_input_x(mul_op)
            .set_attr_axes({-1});
  softmax_op.update_output_desc_y(tensor_desc_softmax_output);

  auto bmm2_op = op::BatchMatMulV2("batchmatmul_v2_02");
  bmm2_op.set_input_x1(softmax_op)
         .set_input_x2(value)
         .set_attr_adj_x1(false)
         .set_attr_adj_x2(false);
  bmm2_op.update_output_desc_y(tensor_bmm2_output);

  auto confusion_trans_op = op::ConfusionTransposeD("confusion_transpose_01");
  confusion_trans_op.set_input_x(bmm2_op)
                    .set_attr_perm({0, 2, 1, 3})
                    .set_attr_shape({32, 16, 384, 64})
                    .set_attr_transpose_first(true);
  confusion_trans_op.update_output_desc_y(tensor_confusion_output);

  std::vector<Operator> inputs{query, key, value, pad_mask, scale, drop_mask};
  std::vector<Operator> outputs{confusion_trans_op};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  //fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("ZAttentionScoreFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  bool findAiCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "AttentionScore") {
      findAiCoreNode = true;
    }
  }
  EXPECT_EQ(findAiCoreNode, true);
}
