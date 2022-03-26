/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file conv_add_fusion_pass.cpp
 * \brief conv-add fusion pass(conv + add --> conv)
 */
#include "conv_add_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "error_util.h"
#include "fp16_t.hpp"
#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const char kPatternConv[] = "Conv";
static const char kPatternAdd[] = "ConvAdd";
static const char kConvolution3D[] = "Conv3D";
static const char kConvolution2D[] = "Conv2D";
static const char kAdd[] = "Add";
static const std::set<string> kNewAddIn = {"Const", "Constant", "Mul"};
static const int64_t kChannelWiseDim = 1;
static const int64_t kBiasIndex = 2;

/*!
  * @brief Define conv+add pattern.
  * The graph struct need to adapt is shown as follows:
  *
  *               const  conv
  *                    \  |
  *                      add
  *                       |
  *                     output
  *
  *  Notice: the struct can be captured by
  *          conv + mul pattern
  *  @return vector<FusionPattern*> All valid patterns.
  */

vector<FusionPattern*> ConvAddFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string pass_name = "TBEConvAddFusion";
  FusionPattern *pattern = new (std::nothrow) FusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "TBEConvAddFusion new an object failed."),
                    return patterns);

  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(kPatternAdd, {kAdd})
      .AddOpDesc(kPatternConv, {kConvolution3D, kConvolution2D})
      .SetInputs(kPatternAdd, {kPatternConv})
      .SetOutput(kPatternAdd);
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  return patterns;
}

/*!
  * @brief Check shape info
  * @param add_node_desc The add node description
  * @param conv_node_desc The conv node description
  * @param non_const_add_input_index The input index of add node from conv node
  * @param channel_for_scalar channel num for scalar
  * @return success for valid shape info
  */
Status ConvAddFusionPass::ShapeInfoCheck(OpDescPtr add_node_desc, OpDescPtr conv_node_desc,
                                         size_t non_const_add_input_index, int64_t& channel_for_scalar) {
  GeTensorDesc non_const_input_desc = add_node_desc->GetInputDesc(non_const_add_input_index);
  GeTensorDesc const_input_desc = add_node_desc->GetInputDesc(1 - non_const_add_input_index);
  auto non_const_dims = non_const_input_desc.GetShape().GetDims();
  auto const_dims = const_input_desc.GetShape().GetDims();

  if (conv_node_desc->GetType() == "Conv2D") {
    auto non_const_input_format = non_const_input_desc.GetFormat();
    int64_t non_const_input_channel;

    if (non_const_dims.size() != 4) {
      OP_LOGW(fused_op_type_.c_str(), "Invalid input size, should be 4.");
      return NOT_CHANGED;
    }

    if (non_const_input_format == FORMAT_NCHW) {
      non_const_input_channel = non_const_dims.at(1);
    } else if (non_const_input_format == FORMAT_NHWC) {
      non_const_input_channel = non_const_dims.at(3);
    } else {
      OP_LOGW(fused_op_type_.c_str(), "Invalid input format, should be NCHW or NHWC.");
      return NOT_CHANGED;
    }
    if (PatternFusionUtil::IsUnknownShape(non_const_input_channel)) {
      OP_LOGW(fused_op_type_.c_str(), "ConvAddFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (const_dims.size() == 0) {
      /* for scalar */
      channel_for_scalar = non_const_input_channel;
    } else if (const_dims.size() == 1) {
      if (PatternFusionUtil::IsUnknownShape(const_dims.at(0))) {
        OP_LOGW(fused_op_type_.c_str(), "ConvAddFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      if (const_dims.at(0) == 1) {
        /* for scalar */
        channel_for_scalar = non_const_input_channel;
      } else if (non_const_input_channel != const_dims.at(0)) {
        /* for vector */
        OP_LOGW(fused_op_type_.c_str(), "Invalid Channel number, %d, %d.", non_const_input_channel, const_dims.at(0));
        return NOT_CHANGED;
      }
    } else if (const_dims.size() == 4) {
      // NHWC, channel wise exemple: add input [1,1,1,256] and conv input [x,x,x,256]
      if (PatternFusionUtil::IsUnknownShape(const_dims.at(0)) or
          PatternFusionUtil::IsUnknownShape(const_dims.at(1)) or
          PatternFusionUtil::IsUnknownShape(const_dims.at(2)) or
          PatternFusionUtil::IsUnknownShape(const_dims.at(3)) or
          PatternFusionUtil::IsUnknownShape(non_const_input_channel)) {
        OP_LOGW(fused_op_type_.c_str(), "ConvAddFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      if ((const_dims.at(0) != 1 or const_dims.at(1) != 1 or
        const_dims.at(2) != 1 or non_const_input_channel != const_dims.at(3)) and
        (non_const_input_format == FORMAT_NHWC)) {
        OP_LOGW(fused_op_type_.c_str(), "Match failed! tensor is not ChannelWise");
        return NOT_CHANGED;
      }

      // NCHW, channel wise exemple: add input [1,256,1,1] and conv input [x,256,x,x]
      if ((const_dims.at(0) != 1 or const_dims.at(1) != non_const_input_channel or
        const_dims.at(2) != 1 or const_dims.at(3) != 1) and
        (non_const_input_format == FORMAT_NCHW)) {
        OP_LOGW(fused_op_type_.c_str(), "Match failed! tensor is not ChannelWise");
        return NOT_CHANGED;
      }
    }
  } else if (conv_node_desc->GetType() == "Conv3D") {
    return CheckConv3DShapeInfo(non_const_dims, const_dims);
  }

  return SUCCESS;
}

/*!
  * @brief Check conv3d shape info
  * @param non_const_dims non const dim value
  * @param const_dims const dim value
  * @return success for valid shape info
  */
Status ConvAddFusionPass::CheckConv3DShapeInfo(const std::vector<int64_t> &non_const_dims,
                                               const std::vector<int64_t> &const_dims) {
  if (const_dims.size() == 1) {
    if (PatternFusionUtil::IsUnknownShape(const_dims.at(0)) ||
        PatternFusionUtil::IsUnknownShape(non_const_dims.at(4))) {
      OP_LOGW(fused_op_type_.c_str(), "ConvAddFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (non_const_dims.at(4) != const_dims.at(0)) {
      return NOT_CHANGED;
    }
  } else if (const_dims.size() == 5) {
    // NDHWC, channel wise  exemple: add input [1,1,1,1,256] and conv input [x,x,x,x,256]
    if (PatternFusionUtil::IsUnknownShape(const_dims.at(0)) || PatternFusionUtil::IsUnknownShape(const_dims.at(1)) ||
        PatternFusionUtil::IsUnknownShape(const_dims.at(2)) || PatternFusionUtil::IsUnknownShape(const_dims.at(3)) ||
        PatternFusionUtil::IsUnknownShape(const_dims.at(4)) ||
        PatternFusionUtil::IsUnknownShape(non_const_dims.at(4))) {
      OP_LOGW(fused_op_type_.c_str(), "ConvAddFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (const_dims.at(0) != 1 or const_dims.at(1) != 1 or const_dims.at(2) != 1 or const_dims.at(3) != 1 or
        non_const_dims.at(4) != const_dims.at(4)) {
      OP_LOGW(fused_op_type_.c_str(), "Match failed! tensor is not ChannelWise");
      return NOT_CHANGED;
    }
  }

  return SUCCESS;
}

/*!
  * @brief connect the conv node to output node
  * @param conv_node The pointer of conv node
  * @param add_node The pointer of add node
  * @return success for connet conv node to output node
  */
Status ConvAddFusionPass::ConnectConvToOutput(NodePtr conv_node, NodePtr add_node) {
  auto conv_node_anchor = conv_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(conv_node_anchor == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "conv node anchor is null"), return FAILED);
  auto add_node_anchor = add_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(add_node_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "add node anchor is null"),
                    return FAILED);

  OutDataAnchor::Vistor<InDataAnchorPtr> output_in_anchors = add_node_anchor->GetPeerInDataAnchors();
  add_node_anchor->UnlinkAll();

  for (auto in_anchor : output_in_anchors) {
    FUSION_PASS_CHECK(in_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "output node anchor is null"),
                      return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(conv_node_anchor, in_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from conv to output failed"),
                      return FAILED);
  }

  return SUCCESS;
}

/*!
  * @brief set node data by bias
  * @param y_data The new data
  * @param channel_for_scalar The channel num
  * @param conv_bias_ptr The pointer of bias tensor
  * @return success for set output data
  */
template <typename T>
void SetOutputData(std::vector<T>& y_data, int64_t channel_for_scalar, GeTensorPtr conv_bias_ptr) {
  uint8_t *aligned_addr = conv_bias_ptr->MutableData().data();
  if (aligned_addr != nullptr) {
    const T* datak = reinterpret_cast<const T*>(aligned_addr);
    if (datak != nullptr) {
      for (int64_t i = 0; i < channel_for_scalar; i++) {
        y_data.push_back(datak[0]);
      }
    }
  }
}

/*!
  * @brief broadcast the scalar
  * @param channel_for_scalar The channel num
  * @param new_shape The new channel num
  * @param conv_bias_ptr The pointer of bias tensor
  * @return success for broadcast the scalar
  */
Status ConvAddFusionPass::BroadcastScalar(int64_t channel_for_scalar, int64_t& new_shape, GeTensorPtr conv_bias_ptr) {
  /* no need broadcast */
  if (channel_for_scalar == 0) {
    return SUCCESS;
  }

  DataType data_type = conv_bias_ptr->GetTensorDesc().GetDataType();
  uint32_t length = 0;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    OP_LOGI(fused_op_type_.c_str(), "Can't GetDataTypeLength of data_type: %s",
            TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }

  std::vector<fp16_t> y_data_fp16_t;
  std::vector<float> y_data_float;

  switch (data_type) {
    case DT_FLOAT16:
      SetOutputData(y_data_fp16_t, channel_for_scalar, conv_bias_ptr);
      conv_bias_ptr->SetData(reinterpret_cast<uint8_t*>(y_data_fp16_t.data()), y_data_fp16_t.size() * length);
      break;
    case DT_FLOAT:
      SetOutputData(y_data_float, channel_for_scalar, conv_bias_ptr);
      conv_bias_ptr->SetData(reinterpret_cast<uint8_t*>(y_data_float.data()), y_data_float.size() * length);
      break;
    default:
      OP_LOGW(fused_op_type_.c_str(), "not support data_type: %s",
              TypeUtils::DataTypeToSerialString(data_type).c_str());
      return NOT_CHANGED;
  }

  new_shape = channel_for_scalar;

  OP_LOGD(fused_op_type_.c_str(), "Broadcast the scalar, new shape is %d.", new_shape);

  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ConvAddFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  OP_LOGI(fused_op_type_.c_str(), "Enter ConvAddFusionPass");
  ge::NodePtr conv_node = GetNodeFromMapping(kPatternConv, mapping);
  ge::NodePtr add_node = GetNodeFromMapping(kPatternAdd, mapping);
  FUSION_PASS_CHECK(conv_node == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "Conv Node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add_node == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "Node Add is null, fusion failed."),
                    return PARAM_INVALID);

  if (conv_node->GetOutDataNodes().size() > 1) {
    OP_LOGI(fused_op_type_.c_str(), "out data size is invalid.");
    return NOT_CHANGED;
  }

  ge::OpDescPtr conv_node_desc = conv_node->GetOpDesc();
  FUSION_PASS_CHECK(conv_node_desc == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "Node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  if ((conv_node_desc->GetType() == "Conv3D") && (add_node->GetOutDataNodes().size() > 1)) {
    OP_LOGI(fused_op_type_.c_str(), "add_node out data size is invalid.");
    return NOT_CHANGED;
  }

  // get conv's weights & bias
  int64_t in_edges_size = conv_node->GetInDataNodes().size();
  OP_LOGI(fused_op_type_.c_str(), "op [%s]: Conv2d has %d input nodes.", conv_node->GetName().c_str(), in_edges_size);
  bool has_bias = !(in_edges_size <= kBiasIndex);

  OutDataAnchorPtr filter_anchor = conv_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(filter_anchor == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "filter output anchor is null"), return PARAM_INVALID);
  NodePtr filter_node = filter_anchor->GetOwnerNode();
  FUSION_PASS_CHECK(filter_node == nullptr,
                    CommonRuntimeErrLog(fused_op_type_.c_str(), "ConvAddFusionPass: filter_node is not exist."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(kNewAddIn.find(filter_node->GetOpDesc()->GetType()) == kNewAddIn.end(),
                    OP_LOGW(fused_op_type_.c_str(), "middle layer's of filter_node is not const type"), return false);

  OutDataAnchorPtr bias_anchor = nullptr;
  NodePtr bias_node = nullptr;
  if (has_bias) {
    bias_anchor = conv_node->GetInDataAnchor(kBiasIndex)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(bias_anchor == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "bias anchor is null"),
                      return PARAM_INVALID);
    bias_node = bias_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(bias_node == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "bias_node is not exist."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(kNewAddIn.find(bias_node->GetOpDesc()->GetType()) == kNewAddIn.end(),
                      OP_LOGW(fused_op_type_.c_str(), "middle layer's of bias_node is not const type"), return false);
  }

  vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(add_node);
  auto bias_add_weight_size = weights.size();
  if (bias_add_weight_size == 0 && add_node->GetType() == kAdd &&
      (conv_node->GetType() == kConvolution3D || conv_node->GetType() == kConvolution2D)) {
    bool check_null = add_node->GetInDataAnchor(1) == nullptr ||
                      add_node->GetInDataAnchor(1)->GetPeerOutAnchor() == nullptr ||
                      add_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode() == nullptr;

    FUSION_PASS_CHECK(check_null,
                      OP_LOGI(fused_op_type_.c_str(), "The input of add %s is null!", add_node->GetName().c_str()),
                      return NOT_CHANGED);
    auto node_in_front_of_add = add_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
    FUSION_PASS_CHECK((conv_node->GetType() == kConvolution2D || node_in_front_of_add->GetType() == "Reshape"),
                       OP_LOGI(fused_op_type_.c_str(), "add other input is %s , not changed.",
                               node_in_front_of_add->GetName().c_str()),
                       return NOT_CHANGED);
    if (node_in_front_of_add->GetType() == "Reshape") {
      /* This case, the BiasAdd is Add and the input of Add is Reshape,
       * we just get the first weight of reshape as the bias. */
      weights = ge::OpDescUtils::GetWeights(node_in_front_of_add);
      FUSION_PASS_CHECK(weights.empty(),
                        OP_LOGI(fused_op_type_.c_str(), "Node Add:[%s]'s weight size %u is invalid.",
                                node_in_front_of_add->GetName().c_str(), weights.size()),
                        return NOT_CHANGED);
    } else {
      OP_LOGI(fused_op_type_.c_str(), "The input of biasadd %s is invalid.", add_node->GetName().c_str());
      return NOT_CHANGED;
    }
  } else {
    FUSION_PASS_CHECK(bias_add_weight_size != 1,
                      OP_LOGI(fused_op_type_.c_str(), "Node BiasAdd:[%s]'s weight size %u is invalid.",
                              add_node->GetName().c_str(), bias_add_weight_size),
                      return NOT_CHANGED);
    /* The weights will be the weight of BiasAdd node */
  }

  ge::OpDescPtr add_node_desc = add_node->GetOpDesc();
  FUSION_PASS_CHECK(
      add_node_desc == nullptr,
      CommonRuntimeErrLog(fused_op_type_.c_str(), "Node's OpDesc is null, fusion failed."),
      return PARAM_INVALID);

  size_t non_const_add_input_index = 0;
  FUSION_PASS_CHECK(true != ge::OpDescUtils::GetNonConstInputIndex(add_node, 0, non_const_add_input_index),
                    OP_LOGW(fused_op_type_.c_str(), "get add's non-const input failed"), return NOT_CHANGED);
  FUSION_PASS_CHECK(((non_const_add_input_index != 0) && (non_const_add_input_index != 1)),
                    OP_LOGW(fused_op_type_.c_str(), "Invalid non-const index"), return NOT_CHANGED);

  int64_t channel_for_scalar = 0;
  FUSION_PASS_CHECK(ConvAddFusionPass::ShapeInfoCheck(add_node_desc, conv_node_desc,
                                                       non_const_add_input_index, channel_for_scalar) != SUCCESS,
                    OP_LOGW(fused_op_type_.c_str(), "check shape info failed"), return NOT_CHANGED);

  GeTensorDesc conv_bias_tensor;
  if (has_bias) {
    // unlink bias to conv
    InDataAnchorPtr conv_bias_in_anchor = conv_node->GetInDataAnchor(kBiasIndex);
    FUSION_PASS_CHECK(conv_bias_in_anchor == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "conv_node bias anchor is null"), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(bias_anchor, conv_bias_in_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "remove edge from input to conv2d failed"),
                      return FAILED);

    // get add_const_node
    vector<ge::NodePtr> add_const_nodes = ge::OpDescUtils::GetConstInputs(add_node);
    FUSION_PASS_CHECK(add_const_nodes.size() <= 0,
                      OP_LOGI(fused_op_type_.c_str(), " add const_node's size %u is invalid.", add_const_nodes.size()),
                      return NOT_CHANGED);
    NodePtr add_const_node = add_const_nodes[0];
    OP_LOGI(fused_op_type_.c_str(), "add_const_nod_dtype[%s]: [%s]", add_const_node->GetName().c_str(),
            add_const_node->GetType().c_str());
    bool check_null = add_node->GetInDataAnchor(1 - non_const_add_input_index) == nullptr ||
                      add_node->GetInDataAnchor(1 - non_const_add_input_index)->GetPeerOutAnchor() == nullptr ||
                      add_node->GetInDataAnchor(1 - non_const_add_input_index)->GetPeerOutAnchor()->GetOwnerNode()
                      == nullptr;
    FUSION_PASS_CHECK(check_null,
                      OP_LOGI(fused_op_type_.c_str(), "The input of add %s is null!", add_node->GetName().c_str()),
                      return NOT_CHANGED);
    OutDataAnchorPtr const_add_anchor = add_node->GetInDataAnchor(1 - non_const_add_input_index)->GetPeerOutAnchor();
    // copy add opDesc, update input_shape & out_shape
    OpDescPtr bias_add_node_desc = AttrUtils::CloneOpDesc(add_node_desc);
    FUSION_PASS_CHECK(bias_add_node_desc == nullptr,
                      OP_LOGI(fused_op_type_.c_str(), "Node's OpDesc is null, clone bias add desc failed."),
                      return PARAM_INVALID);
    OP_LOGD(fused_op_type_.c_str(), "bias_add_node_desc %s, optye %s, input %ld, output %ld",
            bias_add_node_desc->GetName().c_str(), bias_add_node_desc->GetType().c_str(),
            bias_add_node_desc->GetAllInputsDesc().size(), bias_add_node_desc->GetAllOutputsDesc().size());
    GeTensorDesc conv_bias_const_tensor = bias_node->GetOpDesc()->GetOutputDesc(0);
    conv_bias_const_tensor.SetOriginShape(conv_bias_const_tensor.GetShape());
    conv_bias_const_tensor.SetOriginDataType(conv_bias_const_tensor.GetDataType());
    conv_bias_const_tensor.SetOriginFormat(conv_bias_const_tensor.GetOriginFormat());
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's bias_node input datatype is %d.", conv_bias_const_tensor.GetDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's bias_node origin datatype is %d.",
            conv_bias_const_tensor.GetOriginDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's bias_node input format is %d.", conv_bias_const_tensor.GetFormat());
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's bias_node input origin format is %d.",
            conv_bias_const_tensor.GetOriginFormat());
    conv_bias_tensor = conv_node_desc->GetInputDesc(kBiasIndex);
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's 3nd input datatype is %d.", conv_bias_tensor.GetDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's 3nd origin datatype is %d.", conv_bias_tensor.GetOriginDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's 3nd input format is %d.", conv_bias_tensor.GetFormat());
    OP_LOGI(fused_op_type_.c_str(), "Conv2D's 3nd input origin format is %d.", conv_bias_tensor.GetOriginFormat());
    conv_bias_tensor.SetOriginShape(conv_bias_tensor.GetShape());
    conv_bias_tensor.SetOriginDataType(conv_bias_tensor.GetDataType());
    conv_bias_tensor.SetOriginFormat(conv_bias_tensor.GetOriginFormat());
    bias_add_node_desc->SetName(add_node->GetName() + "_bias");
    bias_add_node_desc->SetType(kAdd);
    bias_add_node_desc->UpdateInputName(add_node_desc->GetAllInputName());
    bias_add_node_desc->UpdateOutputName(add_node_desc->GetAllOutputName());
    bias_add_node_desc->UpdateInputDesc(non_const_add_input_index, conv_bias_const_tensor);
    bias_add_node_desc->UpdateOutputDesc(0, conv_bias_tensor);
    if (bias_add_node_desc->HasAttr(ge::ATTR_NAME_BATCH_LABEL)) {
      bias_add_node_desc->DelAttr(ge::ATTR_NAME_BATCH_LABEL);
    }

    // add edge for bias->biasMul
    ge::NodePtr bias_mul_op_node = graph.AddNode(bias_add_node_desc);
    FUSION_PASS_CHECK(bias_mul_op_node == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add bnias mul op node failed"), return FAILED);
    new_nodes.push_back(bias_mul_op_node);
    InDataAnchorPtr bias_mul0_anchor = bias_mul_op_node->GetInDataAnchor(non_const_add_input_index);
    FUSION_PASS_CHECK(bias_mul0_anchor == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "bias_mul input_0 anchor is null"), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(bias_anchor, bias_mul0_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from bias to bias_mul failed"),
                      return FAILED);
    InDataAnchorPtr bias_add1_anchor = bias_mul_op_node->GetInDataAnchor(1 - non_const_add_input_index);
    FUSION_PASS_CHECK(bias_add1_anchor == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "bias_mul input_1 anchor is null"), return FAILED);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(const_add_anchor, bias_add1_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from bias to bias_mul failed"),
                      return FAILED);
    OutDataAnchorPtr bias_mul_out_anchor = bias_mul_op_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(bias_mul_out_anchor, conv_bias_in_anchor) != GRAPH_SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "add edge from bias_mul to conv failed"),
                      return FAILED);
    FUSION_PASS_CHECK(ConvAddFusionPass::ConnectConvToOutput(conv_node, add_node) != SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "connect conv node to output failed"), return FAILED);
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(add_node),
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Remove node failed"), return FAILED);
  } else {
    OP_LOGI(fused_op_type_.c_str(), "ConvAddFusionPass: Conv3d [%s] has %u input anchor.", conv_node->GetName().c_str(),
            conv_node->GetAllInDataAnchors().size());
    OP_LOGI(fused_op_type_.c_str(), "ConvAddFusionPass: Conv3d [%s] has %u input desc.", conv_node->GetName().c_str(),
            conv_node_desc->GetAllInputsDesc().size());

    ge::ConstGeTensorPtr biases = weights[0];
    FUSION_PASS_CHECK(biases == nullptr,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Biasadd node's weight is null, fusion failed."),
                      return PARAM_INVALID);

    int64_t new_shape = 1;
    auto biase_dims_size = biases->GetTensorDesc().GetShape().GetDims().size();
    FUSION_PASS_CHECK((biase_dims_size == 0) && (channel_for_scalar == 0),
                      OP_LOGI(fused_op_type_.c_str(), "Dims size is invalid."), return NOT_CHANGED);

    if (biase_dims_size > 1) {
      for (int64_t dim : biases->GetTensorDesc().GetShape().GetDims()) {
        if (PatternFusionUtil::IsUnknownShape(dim)) {
          OP_LOGW(fused_op_type_.c_str(), "ConvAddFusionPass cannot be applied for unknown shape.");
          return NOT_CHANGED;
        }
        if (dim != kChannelWiseDim) {
          new_shape = dim;
          break;
        }
      }
    } else if (biase_dims_size == 1) {
      new_shape = biases->GetTensorDesc().GetShape().GetDims().at(0);
    }

    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::LinkControlEdge(add_node, conv_node),
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Link control edge failed"), return FAILED);

    ge::GeTensorPtr conv_bias_ptr = nullptr;
    conv_bias_ptr = std::make_shared<ge::GeTensor>(biases->Clone());

    FUSION_PASS_CHECK(
        conv_bias_ptr == nullptr,
        CommonRuntimeErrLog(fused_op_type_.c_str(), "Clone Biasadd node's weight is null, fusion failed."),
        return PARAM_INVALID);

    FUSION_PASS_CHECK(BroadcastScalar(channel_for_scalar, new_shape, conv_bias_ptr) != SUCCESS,
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Broadcase the scalar failed."), return NOT_CHANGED);

    std::vector<int64_t> new_dim_vec;
    new_dim_vec.push_back(new_shape);
    ge::GeShape bias_shape(new_dim_vec);
    conv_bias_ptr->MutableTensorDesc().SetShape(bias_shape);
    conv_bias_ptr->MutableTensorDesc().SetOriginShape(bias_shape);
    ge::Format input_format = add_node_desc->GetInputDesc(0).GetFormat();
    conv_bias_ptr->MutableTensorDesc().SetFormat(input_format);
    input_format = add_node_desc->GetInputDesc(0).GetOriginFormat();
    conv_bias_ptr->MutableTensorDesc().SetOriginFormat(input_format);
    OP_LOGD(fused_op_type_.c_str(), "bias's format is %d, origin format is %d.",
            conv_bias_ptr->MutableTensorDesc().GetFormat(), conv_bias_ptr->MutableTensorDesc().GetOriginFormat());

    FUSION_PASS_CHECK(
        PatternFusionUtil::SetWeightByIndex(conv_node, conv_bias_ptr, kBiasIndex, graph) != SUCCESS,
        CommonRuntimeErrLog(conv_node->GetName().c_str(), "Fail to add bias const node for pooling node."),
        return FAILED);
    FUSION_PASS_CHECK(true != ge::AttrUtils::SetBool(conv_node->GetOpDesc(), ge::MATMUL_HAS_BIAS, true),
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Biasadd op weight should be 1-D."), return FAILED);
    conv_bias_tensor = conv_node_desc->GetInputDesc(kBiasIndex);
    GeTensorDesc input_tensor = conv_node_desc->GetInputDesc(0);
    conv_bias_tensor.SetOriginShape(conv_bias_tensor.GetShape());
    conv_bias_tensor.SetOriginDataType(conv_bias_tensor.GetDataType());
    conv_bias_tensor.SetOriginFormat(input_tensor.GetOriginFormat());
    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input datatype is %d.", conv_node_desc->GetInputDesc(2).GetDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input origin datatype is %d.",
            conv_node_desc->GetInputDesc(2).GetOriginDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input format is %d.", conv_node_desc->GetInputDesc(2).GetFormat());
    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input origin format is %d.",
            conv_node_desc->GetInputDesc(2).GetOriginFormat());
    conv_node_desc->UpdateInputDesc(kBiasIndex, conv_bias_tensor);

    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input datatype is %d.", conv_node_desc->GetInputDesc(2).GetDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input origin datatype is %d.",
            conv_node_desc->GetInputDesc(2).GetOriginDataType());
    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input format is %d.", conv_node_desc->GetInputDesc(2).GetFormat());
    OP_LOGI(fused_op_type_.c_str(), "Conv's 2nd input origin format is %d.",
            conv_node_desc->GetInputDesc(2).GetOriginFormat());
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(add_node),
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "Remove node failed"), return FAILED);
    new_nodes.push_back(conv_node);
  }

  OP_LOGI(fused_op_type_.c_str(), "ConvAddFusionPass fusion success.");
  return SUCCESS;
}

REGISTER_PASS("TBEConvAddFusion", BUILT_IN_GRAPH_PASS, ConvAddFusionPass);
}  // namespace fe
