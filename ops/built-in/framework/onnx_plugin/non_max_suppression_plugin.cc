/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file non_max_suppression_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "nn_detect_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
  constexpr int boxes_index = 0;
  constexpr int socre_index = 1;
  constexpr int max_output_boxes_index = 2;
  constexpr int iou_threshold_index = 3;
  constexpr int score_threshold_index = 4;
}

namespace domi {
using NodeProto = ge::onnx::NodeProto;
using NMSOpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsNonMaxSuppression(const Message* op_src, ge::Operator& op_dest)
{
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int center_point_box = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "center_point_box" && attr.type() == ge::onnx::AttributeProto::INT) {
      center_point_box = attr.i();
    }
  }
  op_dest.SetAttr("center_point_box", center_point_box);

  int input_size = node->input_size();
  op_dest.SetAttr("input_size", input_size);
  op_dest.SetAttr("original_type", "ai.onnx::11::NonMaxSuppression");
  NMSOpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "op_desc is null.");
    return FAILED;
  }
  op_desc->AddDynamicInputDesc("x", input_size);
  op_desc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

Status ParseOpToGraphNonMaxSuppression(const ge::Operator& op, Graph& graph)
{
  auto boxes = op::Data("boxes").set_attr_index(boxes_index);
  auto scores = op::Data("scores").set_attr_index(socre_index);
  auto max_output_boxes = op::Data("max_output_boxes").set_attr_index(max_output_boxes_index);
  auto iou_threshold = op::Data("iou_threshold").set_attr_index(iou_threshold_index);
  auto score_threshold = op::Data("score_threshold").set_attr_index(score_threshold_index);

  int input_size = 0;
  if (op.GetAttr("input_size", input_size) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get input_size from op failed.");
    return FAILED;
  }
  int center_point_box = 0;
  if (op.GetAttr("center_point_box", center_point_box) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get center_point_box from op failed.");
    return FAILED;
  }

  auto non_max_suppression = op::NonMaxSuppressionV6();
  std::vector<Operator> inputs{boxes, scores};
  std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
  if (input_size == (socre_index + 1)) {
    non_max_suppression.set_input_boxes(boxes)
                       .set_input_scores(scores)
                       .set_attr_center_point_box(center_point_box);
  } else if (input_size == (max_output_boxes_index + 1)) {
    inputs.push_back(max_output_boxes);
    non_max_suppression.set_input_boxes(boxes)
                       .set_input_scores(scores)
                       .set_input_max_output_size(max_output_boxes)
                       .set_attr_center_point_box(center_point_box);
  } else if (input_size == (iou_threshold_index + 1)) {
    inputs.push_back(max_output_boxes);
    inputs.push_back(iou_threshold);
    non_max_suppression.set_input_boxes(boxes)
                       .set_input_scores(scores)
                       .set_input_max_output_size(max_output_boxes)
                       .set_input_iou_threshold(iou_threshold)
                       .set_attr_center_point_box(center_point_box);
  } else if (input_size == (score_threshold_index + 1)) {
    inputs.push_back(max_output_boxes);
    inputs.push_back(iou_threshold);
    inputs.push_back(score_threshold);
    non_max_suppression.set_input_boxes(boxes)
                       .set_input_scores(scores)
                       .set_input_max_output_size(max_output_boxes)
                       .set_input_iou_threshold(iou_threshold)
                       .set_input_score_threshold(score_threshold)
                       .set_attr_center_point_box(center_point_box);
  } else {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "The input_size is error.");
    return FAILED;
  }

  auto output_int64 = op::Cast("cast").set_input_x(non_max_suppression).set_attr_dst_type(DT_INT64);
  output_indexs.emplace_back(output_int64, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::10::NonMaxSuppression",
                 "ai.onnx::11::NonMaxSuppression",
                 "ai.onnx::12::NonMaxSuppression",
                 "ai.onnx::13::NonMaxSuppression"})
  .ParseParamsFn(ParseParamsNonMaxSuppression)
  .ParseOpToGraphFn(ParseOpToGraphNonMaxSuppression)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
