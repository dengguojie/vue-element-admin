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

class swin_attention_score_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "swin_attention_score_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "swin_attention_score_fusion_pass_test TearDown" << std::endl;
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

TEST_F(swin_attention_score_fusion_pass_test, swin_attention_score_fusion_pass_test_01) {
  std::string testCaseName = "swin_attention_score_fusion_pass_test_01";
  int32_t batch_dim1 = 64;
  int32_t batch_dim2 = 4;
  int32_t seq_num = 9;
  int32_t nz_dim = 16;
  int32_t n_num = 2;

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
  auto pad_mask1 = CreateDataNode("padding_mask1", {1, batch_dim2, seq_num * nz_dim, seq_num * nz_dim}, FORMAT_ND, DT_FLOAT16);
  auto pad_mask2 = CreateDataNode("padding_mask2", {1, batch_dim1, 1, seq_num * nz_dim, seq_num * nz_dim}, FORMAT_ND, DT_FLOAT16);
  auto scale = CreateDataNode("scale", {1}, FORMAT_ND, DT_FLOAT16);
  auto drop_mask = CreateDataNode("drop_mask", {1, batch_dim1, 1, seq_num * nz_dim, seq_num * nz_dim}, FORMAT_ND, DT_FLOAT16);
  auto perm = CreateDataNode("perm", {4}, FORMAT_ND, DT_INT32);
  auto shape1 = CreateDataNode("shape1", {5}, FORMAT_ND, DT_INT32);
  auto shape2 = CreateDataNode("shape2", {4}, FORMAT_ND, DT_INT32);
  auto shape3 = CreateDataNode("shape3", {3}, FORMAT_ND, DT_INT32);

  ge::Shape shape_x1({batch_dim1, batch_dim2, seq_num * nz_dim, seq_num * nz_dim});
  ge::Shape shape_x2({batch_dim1, batch_dim2, seq_num * nz_dim, n_num * nz_dim});
  ge::Shape shape_x3({batch_dim1, batch_dim2, n_num * nz_dim, seq_num * nz_dim});
  ge::Shape shape_reshape({batch_dim1 / 64, 64, batch_dim2, seq_num * nz_dim, seq_num * nz_dim});
  ge::Shape shape_transpose({batch_dim1, seq_num * nz_dim, batch_dim2, n_num * nz_dim});
  ge::Shape shape_reshape2({batch_dim1, batch_dim2, seq_num * nz_dim * n_num * nz_dim});

  ge::TensorDesc tensor_desc_mul_output(shape_x2, FORMAT_ND, DT_FLOAT16);
  tensor_desc_mul_output.SetOriginShape(shape_x2);
  tensor_desc_mul_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_transpose1_output(shape_x3, FORMAT_ND, DT_FLOAT16);
  tensor_desc_transpose1_output.SetOriginShape(shape_x3);
  tensor_desc_transpose1_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_bmm1_output(shape_x1, FORMAT_ND, DT_FLOAT16);
  tensor_desc_bmm1_output.SetOriginShape(shape_x1);
  tensor_desc_bmm1_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_add1_output(shape_x1, FORMAT_ND, DT_FLOAT16);
  tensor_desc_add1_output.SetOriginShape(shape_x1);
  tensor_desc_add1_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_reshape1_output(shape_reshape, FORMAT_ND, DT_FLOAT16);
  tensor_desc_reshape1_output.SetOriginShape(shape_reshape);
  tensor_desc_reshape1_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_add2_output(shape_reshape, FORMAT_ND, DT_FLOAT16);
  tensor_desc_add2_output.SetOriginShape(shape_reshape);
  tensor_desc_add2_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_reshape2_output(shape_x1, FORMAT_ND, DT_FLOAT16);
  tensor_desc_reshape2_output.SetOriginShape(shape_x1);
  tensor_desc_reshape2_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_softmax_output(shape_x1, FORMAT_ND, DT_FLOAT16);
  tensor_desc_softmax_output.SetOriginShape(shape_x1);
  tensor_desc_softmax_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_bmm2_output(shape_x2, FORMAT_ND, DT_FLOAT16);
  tensor_desc_bmm2_output.SetOriginShape(shape_x2);
  tensor_desc_bmm2_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_transpose2_output(shape_transpose, FORMAT_ND, DT_FLOAT16);
  tensor_desc_transpose2_output.SetOriginShape(shape_transpose);
  tensor_desc_transpose2_output.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc tensor_desc_reshape3_output(shape_reshape2, FORMAT_ND, DT_FLOAT16);
  tensor_desc_reshape3_output.SetOriginShape(shape_reshape2);
  tensor_desc_reshape3_output.SetOriginFormat(FORMAT_ND);

  auto mul_op = op::Mul("mul");
  mul_op.set_input_x1(query)
        .set_input_x2(scale);
  mul_op.update_output_desc_y(tensor_desc_mul_output);

  auto transpose1_op = op::Transpose("transpose1");
  transpose1_op.set_input_x(key)
               .set_input_perm(perm);
  transpose1_op.update_output_desc_y(tensor_desc_transpose1_output);

  auto bmm1_op = op::BatchMatMulV2("batchmatmul_v2_01");
  bmm1_op.set_input_x1(mul_op)
         .set_input_x2(transpose1_op)
         .set_attr_adj_x1(false)
         .set_attr_adj_x2(true);
  bmm1_op.update_output_desc_y(tensor_desc_bmm1_output);

  auto add1_op = op::Add("add_01");
  add1_op.set_input_x1(bmm1_op)
         .set_input_x2(pad_mask1);
  add1_op.update_output_desc_y(tensor_desc_add1_output);

  auto reshape1_op = op::Reshape("reshape_01");
  reshape1_op.set_input_x(add1_op)
             .set_input_shape(shape1);
  reshape1_op.update_output_desc_y(tensor_desc_reshape1_output);

  auto add2_op = op::Add("add_02");
  add2_op.set_input_x1(reshape1_op)
         .set_input_x2(pad_mask2);
  add2_op.update_output_desc_y(tensor_desc_add2_output);

  auto reshape2_op = op::Reshape("reshape_02");
  reshape2_op.set_input_x(add2_op)
             .set_input_shape(shape2);
  reshape2_op.update_output_desc_y(tensor_desc_reshape2_output);

  auto softmax_op = op::SoftmaxV2("softmax_v2_01");
  softmax_op.set_input_x(reshape2_op)
            .set_attr_axes({-1});
  softmax_op.update_output_desc_y(tensor_desc_softmax_output);

  auto bmm2_op = op::BatchMatMulV2("batchmatmul_v2_02");
  bmm2_op.set_input_x1(softmax_op)
         .set_input_x2(value)
         .set_attr_adj_x1(false)
         .set_attr_adj_x2(false);
  bmm2_op.update_output_desc_y(tensor_desc_bmm2_output);

  auto transpose2_op = op::Transpose("transpose_02");
  transpose2_op.set_input_x(bmm2_op)
               .set_input_perm(perm);
  transpose2_op.update_output_desc_y(tensor_desc_transpose2_output);

  auto reshape3_op = op::Reshape("reshape3_op");
  reshape3_op.set_input_x(transpose2_op)
             .set_input_shape(shape3);
  reshape3_op.update_output_desc_y(tensor_desc_reshape3_output);

  std::vector<Operator> inputs{query, key, value, pad_mask1, pad_mask2, scale, drop_mask};
  std::vector<Operator> outputs{reshape3_op};

  Graph graph(testCaseName.c_str());
  graph.SetInputs(inputs).SetOutputs(outputs);
  ComputeGraphPtr computeGraph = GraphUtils::GetComputeGraph(graph);

  //fe::FusionPassTestUtils::InferShapeAndType(computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_infer_shape");
  fe::FusionPassTestUtils::RunGraphFusionPass("AAAAAASwinAttentionScoreFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraph);
  GE_DUMP(computeGraph, testCaseName + "_after_fusion");

  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  bool findAiCoreNode = false;
  for (auto iNode : computeGraph->GetAllNodes()) {
    if (iNode->GetType() == "SwinAttentionScore") {
      findAiCoreNode = true;
    }
  }
  EXPECT_EQ(findAiCoreNode, false);
}