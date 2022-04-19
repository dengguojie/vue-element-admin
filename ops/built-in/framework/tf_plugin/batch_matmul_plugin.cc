/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

/*!
 * \file batch_matmul_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>
#include "register/register.h"
#include "common/util/error_manager/error_manager.h"
#include "op_log.h"
#include "../../op_proto/util/axis_util.h"
#include "../../op_proto/util/error_util.h"
#include "graph/operator.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"

namespace domi {
using namespace ge;

Status AutoMappingFnBatchMatMul(const ge::Operator& op_src, ge::Operator& op)
{
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);

  Status ret = AutoMappingByOpFn(op_src, op);
  if (ret != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "tensorflow plugin parser failed.");
    return FAILED;
  }
  bool transposeA = false;
  if (op.GetAttr("adj_x", transposeA) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr adj_x failed.");
    return FAILED;
  }
  bool transposeB = false;
  if (op.GetAttr("adj_y", transposeB) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr adj_y failed.");
    return FAILED;
  }
  op.SetAttr("adj_x1", transposeA);
  op.SetAttr("adj_x2", transposeB);

  ge::AscendString op_type;
  CHECK(op_src.GetOpType(op_type) != ge::GRAPH_SUCCESS, OP_LOGE(op_name.GetString(), "failed to get op_type"),
        return FAILED);
  if (string(op_type.GetString()) != "BatchMatMulV3") {
    OP_LOGI(op_name.GetString(), "op[BatchMatMul] tensorflow plugin parser[AutoMapping] success.");
    return SUCCESS;
  }

  // Set original_type
  op.SetAttr("original_type", "BatchMatMulV3");
  ge::DataType data_type;
  CHECK(op.GetAttr("Tout", data_type) != ge::GRAPH_SUCCESS,
      CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr Tout failed."),
      return FAILED);
  op.SetAttr("dst_type", static_cast<int>(data_type));

  OP_LOGI(op_name.GetString(), "op[BatchMatMulV3] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

static Status ParseOpToGraphBatchMatMulV3(const ge::Operator &op, ge::Graph &graph)
{
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);
  OP_LOGI(op_name.GetString(), "op[BatchMatMulV3] tensorflow plugin ParseOpToGraph start.");
  bool transpose_x1 = false;
  CHECK(op.GetAttr("adj_x1", transpose_x1) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr adj_x1 failed."),
        return FAILED);
  bool transpose_x2 = false;
  CHECK(op.GetAttr("adj_x2", transpose_x2) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr adj_x2 failed."),
        return FAILED);

  ge::Operator data_1 = op::Data("x1").set_attr_index(0);
  ge::Operator data_2 = op::Data("x2").set_attr_index(1);
  std::vector<ge::Operator> inputs{data_1, data_2};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indices;
  auto batch_matmul = op::BatchMatMulV2()
                          .set_input_x1(data_1)
                          .set_input_x2(data_2)
                          .set_attr_adj_x1(transpose_x1)
                          .set_attr_adj_x2(transpose_x2);

  int dst_type;
  CHECK(op.GetAttr("dst_type", dst_type) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr dst_type failed."),
        return FAILED);

  auto cast = op::Cast().set_input_x(batch_matmul).set_attr_dst_type(dst_type);
  output_indices.emplace_back(cast, vector<std::size_t>{0});

  graph.SetInputs(inputs).SetOutputs(output_indices);
  OP_LOGI(op_name.GetString(), "op[BatchMatMulV3] tensorflow plugin ParseOpToGraph success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"BatchMatMul", "BatchMatMulV2"})
    .ParseParamsByOperatorFn(AutoMappingFnBatchMatMul)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"BatchMatMulV3"})
    .ParseParamsByOperatorFn(AutoMappingFnBatchMatMul)
    .ParseOpToGraphFn(ParseOpToGraphBatchMatMulV3)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
