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
 * \file split_conv2d_concat_fusion_pass.cpp
 * \brief convert split+conv2d+concat to group conv2d
 */
#include "split_conv2d_concat_fusion_pass.h"
#include <vector>
#include <sstream>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"

using namespace ge;

namespace fe {

static const char kPatternSplit[] = "split";
static const char kPatternConv2D[] = "conv2d";
static const char kPatternConcatv2[] = "concat_v2";
static const char kConcatv2Type[] = "ConcatV2";
static const char kConcatType[] = "Concat";
static const char kConv2dType[] = "Conv2D";
static const char kSplitType[] = "Split";
static const char kSplitVType[] = "SplitV";
static const char kConstType[] = "Const";
static const char kAttrGroups[] = "groups";
static const char kNameCcatDim[] = "concat_dim";
static const char kCcatHostOp[] = "Concatv2HostCpuOp";
static const char kSptOutKey[] = "y";
static const char kCcatInKey[] = "x";
static const std::set<string> kNewCcatIn = {"Const", "Constant", "QuantBiasOptimization", "QuantWeightRollBack",
                                             "QuantBiasRollBack"};
static const std::set<DataType> kDataTypeIn = {DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32};

/*!
  * @brief Define split+conv2d+concat pattern.
  *
  *             split
  *               /\
  *       /    /       \    \
  * conv2d conv2d ... conv2d conv2d
  *       \    \       /    /
  *               \/
  *            concat(v2)
  *               |
  *
  * @return vector<FusionPattern*> All valid patterns.
  */
vector<FusionPattern*> SplitConv2dConcatPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SplitConv2dConcatPass");
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(kPatternSplit, {kSplitType, kSplitVType})
      .AddOpDesc(kPatternConv2D, {kConv2dType})
      .AddOpDesc(kPatternConcatv2, {kConcatv2Type, kConcatType})
      .SetInputs(kPatternConv2D, {kPatternSplit})
      .SetInputs(kPatternConcatv2, {kPatternConv2D})
      .SetOutput(kPatternConcatv2);
  patterns.push_back(pattern);

  return patterns;
}

/*!
  * @brief Check op type and inputs, when clone a new OpDesc, it's essential to
  *        set right InputName and OuputName.
  * @param spt_output The split output conv2d node list.
  * @param conv_gp_desc New conv2d description.
  * @return bool Whether the middle layer node is all the same.
  */
bool SplitConv2dConcatPass::AnalyzeMidLayer(ge::Node::Vistor<NodePtr>& spt_output, OpDescPtr& conv_gp_desc) {
  NodePtr a_node = spt_output.at(0);
  FUSION_PASS_CHECK(a_node == nullptr, OP_LOGW(fused_op_type_.c_str(), "split out node is null"),
                    return false);
  auto a_input = a_node->GetInDataNodes();
  size_t a_in_cnt = a_input.size();
  FUSION_PASS_CHECK(a_in_cnt < 2, OP_LOGW(fused_op_type_.c_str(), "middle layer's inputs less than 2"), return false);
  OpDescPtr a_desc = a_node->GetOpDesc();
  FUSION_PASS_CHECK(a_desc == nullptr, OP_LOGW(fused_op_type_.c_str(), "split out node desc is null"),
                    return false);
  GeTensorDesc a_weight_tensor = a_desc->GetInputDesc(1);
  vector<int64_t> a_weight_shape = a_weight_tensor.GetOriginShape().GetDims();
  Format a_weight_format = a_weight_tensor.GetOriginFormat();
  for (int64_t ele : a_weight_shape) {
    if (PatternFusionUtil::IsUnknownShape(ele)) {
      OP_LOGW(fused_op_type_.c_str(), "SplitConv2dConcatPass cannot be applied for unknown shape.");
      return false;
    }
  }
  FUSION_PASS_CHECK(a_weight_format != FORMAT_HWCN && a_weight_format != FORMAT_NCHW,
                    OP_LOGW(fused_op_type_.c_str(), "weight format only support HWCN or NCHW"), return false);
  for (auto out_node : spt_output) {
    if (out_node == nullptr) {
      continue;
    }
    std::string types = out_node->GetType();
    FUSION_PASS_CHECK(
        types != kConv2dType,
        OP_LOGW(fused_op_type_.c_str(), "middle layer's type should be %s, not %s", kConv2dType, types.c_str()),
        return false);
    auto inputs = out_node->GetInDataNodes();
    size_t count = inputs.size();
    FUSION_PASS_CHECK(count != a_in_cnt, OP_LOGW(fused_op_type_.c_str(), "middle layer's inputs count is different"),
                      return false);
    auto outputs = out_node->GetOutDataNodes();
    FUSION_PASS_CHECK(outputs.size() != 1, OP_LOGW(fused_op_type_.c_str(), "middle layer's output is multi-refer"),
                      return false);
    FUSION_PASS_CHECK(outputs.at(0) == nullptr, OP_LOGD(fused_op_type_.c_str(), "get node failed"),
                      return false);
    FUSION_PASS_CHECK(outputs.at(0)->GetType() != kConcatv2Type && outputs.at(0)->GetType() != kConcatType,
                      OP_LOGW(fused_op_type_.c_str(), "bottom layer is not ConcatV2 or Concat"), return false);
    for (size_t j = 1; j < count; ++j) {
      FUSION_PASS_CHECK((inputs.at(j) == nullptr || kNewCcatIn.find(inputs.at(j)->GetType()) == kNewCcatIn.end()),
                        OP_LOGW(fused_op_type_.c_str(), "middle layer's other input is not const type"), return false);
    }
    OpDescPtr desc = out_node->GetOpDesc();
    FUSION_PASS_CHECK(desc == nullptr, OP_LOGW(fused_op_type_.c_str(), "out node's desc is null"),
                      return false);
    GeTensorDesc weight_tensor = desc->GetInputDesc(1);
    vector<int64_t> weight_shape = weight_tensor.GetOriginShape().GetDims();
    for (int64_t ele : weight_shape) {
      if (PatternFusionUtil::IsUnknownShape(ele)) {
        OP_LOGW(fused_op_type_.c_str(), "SplitConv2dConcatPass cannot be applied for unknown shape.");
        return false;
      }
    }
    FUSION_PASS_CHECK(weight_shape != a_weight_shape,
                      OP_LOGW(fused_op_type_.c_str(), "middle layer's second input shape is different"), return false);
    Format weight_format = weight_tensor.GetOriginFormat();
    FUSION_PASS_CHECK(weight_format != a_weight_format,
                      OP_LOGW(fused_op_type_.c_str(), "middle layer's second input format is different"),
                      return false);
  }
  conv_gp_desc = AttrUtils::CloneOpDesc(a_desc);
  FUSION_PASS_CHECK(conv_gp_desc == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "clone conv2d desc failed"), return false);
  conv_gp_desc->SetName(a_desc->GetName() + "/group_conv2d");
  auto in_name = a_desc->GetAllInputName();
  auto out_name = a_desc->GetAllOutputName();
  conv_gp_desc->UpdateInputName(in_name);
  conv_gp_desc->UpdateOutputName(out_name);

  return true;
}

/*!
  * @brief Verify split and concat dim is input channel or not.
  *
  * x0     concat_dim   concat_dim   x0
  *  \ ... /                    \    / ...
  *     \/                        \/
  *  concatv2                   concat
  *      |                         |
  *
  * @param conv_gp_desc New conv2d description.
  * @param split_node Split node.
  * @param ccat_node Concat/Concatv2 node.
  * @return bool Whether the verification is passed.
  */
bool SplitConv2dConcatPass::VerifySptCcatAxis(OpDescPtr& conv_desc, NodePtr& split_node, NodePtr& ccat_node) {
  auto spt_inputs = split_node->GetInDataNodes();
  NodePtr spt_const = nullptr;
  if (split_node->GetType() == kSplitVType) {
    spt_const = spt_inputs.at(2);
  } else {
    spt_const = spt_inputs.at(0);
  }
  FUSION_PASS_CHECK(spt_const == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "split input const is null"), return false);
  FUSION_PASS_CHECK(spt_const->GetType() != kConstType,
                    OP_LOGW(fused_op_type_.c_str(), "concat input 0 is not const node"), return false);
  OpDescPtr split_desc = split_node->GetOpDesc();
  FUSION_PASS_CHECK(split_desc == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "split node's desc is null"), return false);
  size_t split_cnt = split_desc->GetOutputsSize();
  size_t ccat_cnt = 0;
  auto ccat_inputs = ccat_node->GetInDataNodes();
  FUSION_PASS_CHECK(split_cnt != ccat_inputs.size() - 1,
                    OP_LOGW(fused_op_type_.c_str(), "split output count is not equal to concat input"), return false);
  for (auto in_node : ccat_inputs) {
    if (in_node->GetType() == kConv2dType) {
      ccat_cnt++;
    }
  }
  FUSION_PASS_CHECK(split_cnt != ccat_cnt, OP_LOGW(fused_op_type_.c_str(), "not all split output is conv2d"),
                    return false);

  vector<ge::GeTensorPtr> spt_weights = ge::OpDescUtils::MutableWeights(spt_const);
  FUSION_PASS_CHECK(spt_weights.size() < 1, OP_LOGW(fused_op_type_.c_str(), "split weights get failed"), return false);
  ge::GeTensorPtr spt_axis_ptr = spt_weights[0];
  FUSION_PASS_CHECK(spt_axis_ptr == nullptr, OP_LOGW(fused_op_type_.c_str(), "split axis is nullptr"), return false);
  int32_t* spt_axis = (int32_t*)spt_axis_ptr->GetData().data();
  FUSION_PASS_CHECK(spt_axis == nullptr, OP_LOGW(fused_op_type_.c_str(), "spt_axis is nullptr"), return false);
  size_t axis_pos = ccat_node->GetType() == kConcatv2Type ? split_cnt : 0;
  NodePtr ccat_const = ccat_inputs.at(axis_pos);
  vector<ge::GeTensorPtr> ccat_weights = ge::OpDescUtils::MutableWeights(ccat_const);
  FUSION_PASS_CHECK(ccat_weights.size() < 1,
                    OP_LOGW(fused_op_type_.c_str(), "concat weights get failed"), return false);
  ge::GeTensorPtr ccar_axis_ptr = ccat_weights[0];
  FUSION_PASS_CHECK(ccar_axis_ptr == nullptr, OP_LOGW(fused_op_type_.c_str(), "concat axis is nullptr"), return false);
  int32_t* ccat_axis = (int32_t*)ccar_axis_ptr->GetData().data();
  FUSION_PASS_CHECK(ccat_axis == nullptr, OP_LOGW(fused_op_type_.c_str(), "ccat_axis is nullptr"), return false);
  FUSION_PASS_CHECK(ccat_axis[0] != spt_axis[0], OP_LOGW(fused_op_type_.c_str(), "split axis is not equal to concat"),
                    return false);

  GeTensorDesc x_tensor = conv_desc->GetInputDesc(0);
  std::string fmt_str = TypeUtils::FormatToSerialString(x_tensor.GetOriginFormat());
  int32_t pos = fmt_str.find('C');
  FUSION_PASS_CHECK(ccat_axis[0] != pos, OP_LOGW(fused_op_type_.c_str(), "split axis is not on input channel"),
                    return false);

  return true;
}

/*!
  * @brief Get split input and concat output desc, update all in/outputs
  *        desc of group conv2d.
  * @param conv_gp_desc New conv2d description.
  * @param split_node Split node.
  * @param ccat_node Concat/Concatv2 node.
  * @return bool Whether the conv2d desc is updated successfully.
  */
bool SplitConv2dConcatPass::UpdateConv2dDesc(OpDescPtr& conv_desc, NodePtr& split_node, NodePtr& ccat_node) {
  FUSION_PASS_CHECK(!VerifySptCcatAxis(conv_desc, split_node, ccat_node),
                    OP_LOGW(fused_op_type_.c_str(), "verify split and concat axis param failed"), return false);
  OpDescPtr split_desc = split_node->GetOpDesc();
  FUSION_PASS_CHECK(split_desc == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "split node's desc is null"), return false);
  GeTensorDesc split_in_tensor = split_desc->GetInputDesc(1);
  if (split_node->GetType() == kSplitVType) {
    split_in_tensor = split_desc->GetInputDesc(0);
  }
  std::vector<int64_t> split_in_shape = split_in_tensor.GetOriginShape().GetDims();
  Format split_format = split_in_tensor.GetOriginFormat();
  OpDescPtr ccat_desc = ccat_node->GetOpDesc();
  FUSION_PASS_CHECK(ccat_desc == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "ccat node's desc is null"), return false);
  GeTensorDesc ccat_out_tensor = ccat_desc->GetOutputDesc(0);
  std::vector<int64_t> ccat_out_shape = ccat_out_tensor.GetOriginShape().GetDims();
  Format ccat_format = ccat_out_tensor.GetOriginFormat();
  FUSION_PASS_CHECK(split_format != ccat_format,
                    OP_LOGW(fused_op_type_.c_str(), "split format is not equal to concat"), return false);
  GeTensorDesc x_tensor = conv_desc->GetInputDesc(0);
  Format x_format = x_tensor.GetOriginFormat();
  FUSION_PASS_CHECK(split_format != x_format, OP_LOGW(fused_op_type_.c_str(), "split format is not equal to conv2d"),
                    return false);
  x_tensor.SetShape(ge::GeShape(split_in_shape));
  x_tensor.SetOriginShape(ge::GeShape(split_in_shape));
  conv_desc->UpdateInputDesc(0, x_tensor);
  GeTensorDesc y_format = conv_desc->GetOutputDesc(0);
  y_format.SetShape(ge::GeShape(ccat_out_shape));
  y_format.SetOriginShape(ge::GeShape(ccat_out_shape));
  conv_desc->UpdateOutputDesc(0, y_format);
  int64_t groups = split_desc->GetOutputsSize();
  FUSION_PASS_CHECK(!AttrUtils::SetInt(conv_desc, kAttrGroups, groups),
                    OP_LOGW(fused_op_type_.c_str(), "set groups attr failed"), return false);
  size_t a_in_cnt = conv_desc->GetInputsSize();
  for (size_t n = 1; n < a_in_cnt; ++n) {
    GeTensorDesc bTensor = conv_desc->GetInputDesc(n);
    DataType bDtype = bTensor.GetDataType();
    FUSION_PASS_CHECK(kDataTypeIn.find(bDtype) == kDataTypeIn.end(),
                      OP_LOGW(fused_op_type_.c_str(),
                              "conv2d %d input data type only support float,"
                              " float16, int8 or int32",
                              int(n)),
                      return false);
    std::vector<int64_t> b_in_shape = bTensor.GetOriginShape().GetDims();
    size_t pos = 0;
    if (b_in_shape.size() == 4) {
      std::string fmt_str = TypeUtils::FormatToSerialString(bTensor.GetOriginFormat());
      size_t found = fmt_str.find('N');
      pos = found == std::string::npos ? 0 : found;
    }
    b_in_shape[pos] *= groups;
    bTensor.SetShape(ge::GeShape(b_in_shape));
    bTensor.SetOriginShape(ge::GeShape(b_in_shape));
    conv_desc->UpdateInputDesc(n, bTensor);
  }

  return true;
}

/*!
  * @brief Create new concat nodes, update it's in/output name and tensor desc.
  *
  * split_dim  input
  *          \/
  *        split
  *          /\
  *  /  /  / ... \  \  \
  * y0  y1 y2   y29 y30 y31
  *
  * x0  x1 x2   x30 x31 concat_dim
  *  \  \  \ ... /  /  /
  *          \/
  *        concatv2
  *           |
  *         output
  *
  * @param split_node Split node.
  * @param ccat_node Concat/Concatv2 node.
  * @param const_desc New Concatv2 nodes.
  * @return bool Whether new Concatv2 desc is added successfully.
  */
bool SplitConv2dConcatPass::AddConcatDesc(NodePtr& split_node, NodePtr& ccat_node, std::vector<OpDescPtr>& const_desc) {
  OpDescPtr ccat_desc = ccat_node->GetOpDesc();
  auto out_name = ccat_desc->GetAllOutputName();

  OpDescPtr spt_desc = split_node->GetOpDesc();
  std::map<std::string, uint32_t> in_name;
  auto spt_out_name = spt_desc->GetAllOutputName();
  for (auto iter : spt_out_name) {
    std::string key = iter.first;
    size_t found = key.find(kSptOutKey);
    if (found == std::string::npos) {
      return false;
    }
    key.replace(found, strlen(kCcatInKey), kCcatInKey);
    in_name.insert(std::make_pair(key, iter.second));
  }
  uint32_t value_dim = spt_out_name.size();
  in_name.insert(std::make_pair(kNameCcatDim, value_dim));

  auto spt_outputs = split_node->GetOutDataNodes();
  size_t count = spt_outputs.size();
  NodePtr a_node = spt_outputs.at(0);
  auto a_input = a_node->GetInDataNodes();
  size_t a_in_cnt = a_input.size();
  for (size_t cout_in = 1; cout_in < a_in_cnt; ++cout_in) {
    OpDescPtr n_desc = AttrUtils::CloneOpDesc(ccat_desc);
    FUSION_PASS_CHECK(n_desc == nullptr, OP_LOGW(fused_op_type_.c_str(), "clone concat desc failed"), return false);
    n_desc->SetType(kCcatHostOp);
    n_desc->UpdateInputName(in_name);
    n_desc->UpdateOutputName(out_name);
    n_desc->DelAttr(ge::ATTR_NO_NEED_CONSTANT_FOLDING);
    NodePtr concat_node = a_input.at(cout_in);
    OpDescPtr concat_desc = concat_node->GetOpDesc();
    GeTensorDesc concat_tensor = concat_desc->GetOutputDesc(0);
    std::vector<int64_t> concat_shape = concat_tensor.GetOriginShape().GetDims();
    Format concat_origin_format = concat_tensor.GetOriginFormat();
    DataType concat_origin_dtype = concat_tensor.GetOriginDataType();
    size_t pos = 0;
    if (concat_shape.size() == 4) {
      std::string fmt_str = TypeUtils::FormatToSerialString(concat_tensor.GetOriginFormat());
      size_t found = fmt_str.find('N');
      pos = found == std::string::npos ? 0 : found;
    }
    for (size_t i = 0; i < count; ++i) {
      GeTensorDesc in_desc = n_desc->GetInputDesc(i);
      in_desc.SetShape(ge::GeShape(concat_shape));
      in_desc.SetOriginShape(ge::GeShape(concat_shape));
      in_desc.SetFormat(concat_origin_format);
      in_desc.SetOriginFormat(concat_origin_format);
      in_desc.SetOriginDataType(concat_origin_dtype);
      in_desc.SetDataType(concat_origin_dtype);
      n_desc->UpdateInputDesc(i, in_desc);
    }
    concat_shape[pos] *= int64_t(count);
    GeTensorDesc out_desc = n_desc->GetOutputDesc(0);
    out_desc.SetShape(ge::GeShape(concat_shape));
    out_desc.SetOriginShape(ge::GeShape(concat_shape));
    out_desc.SetFormat(concat_origin_format);
    out_desc.SetOriginFormat(concat_origin_format);
    out_desc.SetOriginDataType(concat_origin_dtype);
    out_desc.SetDataType(concat_origin_dtype);
    n_desc->UpdateOutputDesc(0, out_desc);
    std::string cout_str = std::to_string(cout_in);
    n_desc->SetName(concat_desc->GetName() + "/concat_" + cout_str);
    const_desc.push_back(n_desc);
    if (ccat_node->GetType() == kConcatType) {
      GeTensorDesc in_desc = n_desc->GetInputDesc(count);
      in_desc.SetShape(ge::GeShape());
      in_desc.SetOriginShape(ge::GeShape());
      in_desc.SetShape(ge::GeShape());
      in_desc.SetFormat(ge::FORMAT_ND);
      in_desc.SetOriginFormat(ge::FORMAT_ND);
      in_desc.SetOriginDataType(ge::DT_INT32);
      in_desc.SetDataType(ge::DT_INT32);
      n_desc->UpdateInputDesc(count, in_desc);
    }
  }

  return true;
}

/*!
  * @brief Create new const_dim node, link conv2d other inputs and new
  *        const_dim to new Concatv2 node.
  *
  * x0  x1 x2   x30 x31 concat_dim
  *  \  \  \ ... /  /  /
  *          \/
  *    Concatv2HostCpuOp  <- [concatv2]
  *           |
  *     ConvBnFilterHost  <- [bn filter bias]
  *           |
  *      GroupPadding     <- [group pad]
  *           |
  *         output
  *
  * @param graph The whole graph.
  * @param split_node Split node.
  * @param const_desc New Concatv2 nodes.
  * @param const_dim concat_dim of New Concatv2 nodes.
  * @return bool Whether conv2d other inputs is linked to new Concatv2 node
  *         successfully.
  */
bool SplitConv2dConcatPass::LinkNewConcat(ge::ComputeGraph& graph, NodePtr& split_node,
                                          std::vector<NodePtr>& const_ccat, std::vector<NodePtr>& const_dim) {
  auto spt_out_anchor = split_node->GetAllOutDataAnchors();
  for (auto out_anchor : spt_out_anchor) {
    int idx = out_anchor->GetIdx();
    InDataAnchorPtr conv_anchor = out_anchor->GetPeerInDataAnchors().at(0);
    FUSION_PASS_CHECK(conv_anchor == nullptr,
                      OP_LOGW(fused_op_type_.c_str(), "get in data anchor failed"), return false);
    NodePtr conv_node = conv_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(conv_node == nullptr,
                      OP_LOGW(fused_op_type_.c_str(), "get input node failed"), return false);
    size_t pos = 0;
    for (auto new_ccat : const_ccat) {
      InDataAnchorPtr new_ccat_in = new_ccat->GetInDataAnchor(idx);
      auto conv_in_anchor = conv_node->GetInDataAnchor(++pos);
      FUSION_PASS_CHECK(conv_in_anchor == nullptr,
                        OP_LOGW(fused_op_type_.c_str(), "get in data anchor failed"), return false);
      auto conv_out_anchor = conv_in_anchor->GetPeerOutAnchor();
      FUSION_PASS_CHECK(conv_out_anchor == nullptr,
                        OP_LOGW(fused_op_type_.c_str(), "get out anchor failed"), return false);
      conv_out_anchor->Unlink(conv_in_anchor);
      Status add_res = GraphUtils::AddEdge(conv_out_anchor, new_ccat_in);
      FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                        OP_LOGW(fused_op_type_.c_str(), "add edge from conv2d other input failed"), return false);
    }
  }
  NodePtr axis_node = nullptr;
  if (split_node->GetType() == kSplitVType) {
    axis_node = split_node->GetInDataNodes().at(2);
  } else {
    axis_node = split_node->GetInDataNodes().at(0);
  }
  FUSION_PASS_CHECK(axis_node == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "get axis node failed"), return false);
  OpDescPtr axis_desc = axis_node->GetOpDesc();
  FUSION_PASS_CHECK(axis_desc == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "get axis node's desc failed"), return false);
  size_t count = 0;
  for (auto new_ccat : const_ccat) {
    OpDescPtr ccat_desc = new_ccat->GetOpDesc();
    FUSION_PASS_CHECK(ccat_desc == nullptr,
                      OP_LOGW(fused_op_type_.c_str(), "get new ccat node's desc failed"), return false);
    int idx = ccat_desc->GetInputIndexByName(kNameCcatDim);
    InDataAnchorPtr last_in_anchor = new_ccat->GetInDataAnchor(idx);
    OpDescPtr new_axis_desc = AttrUtils::CloneOpDesc(axis_desc);
    FUSION_PASS_CHECK(new_axis_desc == nullptr,
                      OP_LOGW(fused_op_type_.c_str(), "clone split input split_dim desc failed"), return false);
    std::string cout_str = std::to_string(count++);
    new_axis_desc->SetName(axis_desc->GetName() + "/last_" + cout_str);
    NodePtr new_axis_node = graph.AddNode(new_axis_desc);
    FUSION_PASS_CHECK(new_axis_node == nullptr,
                      OP_LOGW(fused_op_type_.c_str(), "add new axis node failed"), return false);

    GeTensorDesc ccat_tensor = ccat_desc->GetInputDesc(0);
    std::vector<int64_t> ccat_in_shape = ccat_tensor.GetOriginShape().GetDims();
    int32_t pos = 0;
    if (ccat_in_shape.size() == 4) {
      std::string fmt_str = TypeUtils::FormatToSerialString(ccat_tensor.GetOriginFormat());
      size_t found = fmt_str.find('N');
      pos = found == std::string::npos ? 0 : found;
    }
    std::vector<int32_t> axis;
    axis.push_back(pos);
    vector<ge::GeTensorPtr> axis_weights = ge::OpDescUtils::MutableWeights(new_axis_node);
    FUSION_PASS_CHECK(axis_weights.size() < 1,
                      OP_LOGW(fused_op_type_.c_str(), "axis weights get failed"), return false);
    ge::GeTensorPtr axis_ptr = axis_weights[0];
    FUSION_PASS_CHECK(axis_ptr == nullptr, OP_LOGW(fused_op_type_.c_str(), "axis ptr is nullptr"), return false);
    axis_ptr->SetData(reinterpret_cast<uint8_t*>(axis.data()), axis.size() * sizeof(int32_t));
    Status add_res = GraphUtils::AddEdge(new_axis_node->GetOutDataAnchor(0), last_in_anchor);
    FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                      OP_LOGW(fused_op_type_.c_str(), "add edge from const dim node to new concat failed"),
                      return false);
    const_dim.push_back(new_axis_node);
  }

  return true;
}

/*!
  * @brief Link group conv2d with split previous and concat next node.
  * @param group_conv New conv2d node.
  * @param split_node Split node.
  * @param ccat_node Concat/Concatv2 node.
  * @param const_ccat New Concatv2 nodes.
  * @return bool Whether new conv2d node is linked successfully.
  */
bool SplitConv2dConcatPass::LinkGroupConv2d(NodePtr& group_conv, NodePtr& split_node, NodePtr& ccat_node,
                                            std::vector<NodePtr>& const_ccat) {
  auto in_anchor = split_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(in_anchor == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "split input data anchor is null"), return false);
  OutDataAnchorPtr pre_anchor = in_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(pre_anchor == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "split input anchor is null"), return false);
  pre_anchor->Unlink(split_node->GetInDataAnchor(1));
  InDataAnchorPtr x_anchor = group_conv->GetInDataAnchor(0);
  FUSION_PASS_CHECK(x_anchor == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "group conv2d input anchor is null"), return false);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(pre_anchor, x_anchor) != GRAPH_SUCCESS,
                    OP_LOGW(fused_op_type_.c_str(), "add edge from split input to conv2d failed"), return false);
  OutDataAnchorPtr ccat_out = ccat_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(ccat_out == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "concat output anchor is null"), return false);
  OutDataAnchorPtr y_anchor = group_conv->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(y_anchor == nullptr, OP_LOGW(fused_op_type_.c_str(), "group conv2d output anchor is null"),
                    return false);
  auto peer_in_anchor = ccat_out->GetPeerInDataAnchors();
  for (auto next_anchor : peer_in_anchor) {
    ccat_out->Unlink(next_anchor);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(y_anchor, next_anchor) != GRAPH_SUCCESS,
                      OP_LOGW(fused_op_type_.c_str(), "add edge from conv2d output to concat next failed"),
                      return false);
  }
  size_t pos = 0;
  for (auto new_ccat : const_ccat) {
    Status add_res = GraphUtils::AddEdge(new_ccat->GetOutDataAnchor(0), group_conv->GetInDataAnchor(++pos));
    FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                      OP_LOGW(fused_op_type_.c_str(), "add edge from conv2d to new concat failed"), return false);
  }

  return true;
}

/*!
  * @brief Weight or bias input should be concat by new concat node, while
  *        input image data leave as is, a conv2d node should be created
  *        with new shape and groups attr, after new conv2d node linked in
  *        the graph, other nodes should be deleted.
  *
  *       weight...weight    bias(if exist)
  *           \  |  /        /
  * inputs   concatv2    concatv2
  *    \        |        /
  *     conv2d(with group)
  *             |
  *
  * @param graph The whole graph.
  * @param mapping Matched nodes of defined pattern.
  * @param new_nodes New nodes added to graph.
  * @return Status Graph processing result.
  */
Status SplitConv2dConcatPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<NodePtr>& new_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Enter SplitConv2dConcatPass.");
  NodePtr split_node = GetNodeFromMapping(kPatternSplit, mapping);
  NodePtr ccat_node = GetNodeFromMapping(kPatternConcatv2, mapping);
  FUSION_PASS_CHECK(split_node == nullptr, OP_LOGW(fused_op_type_.c_str(), "Split Node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(ccat_node == nullptr, OP_LOGW(fused_op_type_.c_str(), "Ccat Node is null, fusion failed."),
                    return PARAM_INVALID);
  auto spt_output = split_node->GetOutDataNodes();

  OpDescPtr conv_gp_desc = nullptr;
  FUSION_PASS_CHECK(!AnalyzeMidLayer(spt_output, conv_gp_desc),
                    OP_LOGW(fused_op_type_.c_str(), "nothing changed on the graph"), return NOT_CHANGED);

  FUSION_PASS_CHECK(!UpdateConv2dDesc(conv_gp_desc, split_node, ccat_node),
                    OP_LOGW(fused_op_type_.c_str(), "update group conv2d desc failed"), return NOT_CHANGED);
  NodePtr group_conv = graph.AddNode(conv_gp_desc);
  FUSION_PASS_CHECK(group_conv == nullptr, OP_LOGW(fused_op_type_.c_str(), "add group conv2d node failed"),
                    return FAILED);
  new_nodes.push_back(group_conv);

  std::vector<OpDescPtr> const_desc;
  FUSION_PASS_CHECK(!AddConcatDesc(split_node, ccat_node, const_desc),
                    OP_LOGW(fused_op_type_.c_str(), "create conv2d other input concat desc failed"), return NOT_CHANGED);
  std::vector<NodePtr> const_ccat;
  for (auto new_desc : const_desc) {
    NodePtr new_ccat = graph.AddNode(new_desc);
    FUSION_PASS_CHECK(new_ccat == nullptr, OP_LOGW(fused_op_type_.c_str(), "add new ccat node failed"),
                      return FAILED);
    const_ccat.push_back(new_ccat);
    new_nodes.push_back(new_ccat);
  }

  std::vector<NodePtr> const_dim;
  FUSION_PASS_CHECK(!LinkNewConcat(graph, split_node, const_ccat, const_dim),
                    OP_LOGW(fused_op_type_.c_str(), "create concat last const node failed"), return FAILED);
  for (auto new_last : const_dim) {
    new_nodes.push_back(new_last);
  }

  FUSION_PASS_CHECK(!LinkGroupConv2d(group_conv, split_node, ccat_node, const_ccat),
                    OP_LOGW(fused_op_type_.c_str(), "link group conv2d node and new nodes failed"), return FAILED);

  auto spt_inputs = split_node->GetInDataNodes();
  FUSION_PASS_CHECK(graph.RemoveNode(split_node) != GRAPH_SUCCESS,
                    OP_LOGW(fused_op_type_.c_str(), "remove split node failed"), return FAILED);
  for (auto ccatInNode : ccat_node->GetInDataNodes()) {
    if (ccatInNode->GetType() == kConv2dType) {
      FUSION_PASS_CHECK(graph.RemoveNode(ccatInNode) != GRAPH_SUCCESS,
                        OP_LOGW(fused_op_type_.c_str(), "remove unused conv2d node failed"), return FAILED);
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(ccat_node) != GRAPH_SUCCESS,
                    OP_LOGW(fused_op_type_.c_str(), "remove concat node failed"), return FAILED);
  OP_LOGD(fused_op_type_.c_str(), "Leave SplitConv2dConcatPass.");

  return SUCCESS;
}

REGISTER_PASS("ASplitConv2dConcatPass", BUILT_IN_GRAPH_PASS, SplitConv2dConcatPass);
}  // namespace fe
