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
 * \file conv2d_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "register/register.h"
#include "operator.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/error_manager/error_manager.h"

namespace domi {

/*!
  * @brief Replace GE ParseParams fuction to process graph conv2d node attrs
  * @param op_src the source op info from onnx.
  * @param op the dest GE op.
  * @return status whether this operation success.
  */
Status ParseParamsConv2D(const Message* op_src, ge::Operator& op) {
  // Convert original onnx graph conv attrs to GE graph attrs
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "Dynamic cast op_src to NodeProto failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }

  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "get op desc failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }

  // The fmap should be NCHW
  ge::GeTensorDesc org_tensor_x = op_dsc->GetInputDesc(0);
  org_tensor_x.SetOriginFormat(ge::FORMAT_NCHW);
  org_tensor_x.SetFormat(ge::FORMAT_NCHW);
  auto ret_x = op_dsc->UpdateInputDesc(0, org_tensor_x);
  if (ret_x != ge::GRAPH_SUCCESS) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "update fmap format failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }

  // The filter should be NCHW
  ge::GeTensorDesc org_tensor_w = op_dsc->GetInputDesc(1);
  org_tensor_w.SetOriginFormat(ge::FORMAT_NCHW);
  org_tensor_w.SetFormat(ge::FORMAT_NCHW);
  auto ret_w = op_dsc->UpdateInputDesc(1, org_tensor_w);
  if (ret_w != ge::GRAPH_SUCCESS) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "update filter format failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }

  // The output should be NCHW
  ge::GeTensorDesc org_tensor_y = op_dsc->GetOutputDesc(0);
  org_tensor_y.SetOriginFormat(ge::FORMAT_NCHW);
  org_tensor_y.SetFormat(ge::FORMAT_NCHW);
  auto ret_y = op_dsc->UpdateOutputDesc(0, org_tensor_y);
  if (ret_y != ge::GRAPH_SUCCESS) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["description"] = "update output format failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  }

  // set default value here
  // if attr is set in onnx model, then update

  // set default strides
  std::vector<int32_t> strides_list_default = {1, 1, 1, 1};
  op.SetAttr("strides", strides_list_default);

  // set default dilations
  std::vector<int32_t> dilations_list_default = {1, 1, 1, 1};
  op.SetAttr("dilations", dilations_list_default);

  // set default pads
  std::vector<int32_t> pad_list_default = {0, 0, 0, 0};
  op.SetAttr("pads", pad_list_default);

  // set default groups
  int32_t groups_default = 1;
  op.SetAttr("groups", groups_default);

  // set default auto_pad
  std::string auto_pad_default = "NOTSET";
  op.SetAttr("auto_pad", auto_pad_default);

  // set data_format, set to NCHW
  op.SetAttr("data_format", ge::FORMAT_NCHW);

  // if attr is set in model, receive them with these var
  std::vector<int32_t> strides_list;
  std::vector<int32_t> dilations_list;
  std::vector<int32_t> pad_list;

  // update attrs with model value
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 2) {
        strides_list.push_back(1);
        strides_list.push_back(1);
        strides_list.push_back(attr.ints(0));
        strides_list.push_back(attr.ints(1));
      }
      op.SetAttr("strides", strides_list);
    } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 2) {
        dilations_list.push_back(1);
        dilations_list.push_back(1);
        dilations_list.push_back(attr.ints(0));
        dilations_list.push_back(attr.ints(1));
      }
      op.SetAttr("dilations", dilations_list);
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      // in onnx pads=[top, left, bottomm, right] -> [top, bottom, left, right]
      if (attr.ints_size() == 4) {
        pad_list.push_back(attr.ints(0));
        pad_list.push_back(attr.ints(2));
        pad_list.push_back(attr.ints(1));
        pad_list.push_back(attr.ints(3));
      }
      op.SetAttr("pads", pad_list);
    } else if (attr.name() == "group" && attr.type() == ge::onnx::AttributeProto::INT) {
      op.SetAttr("groups", attr.i());
    } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      op.SetAttr("auto_pad", attr.s());
    }
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2D")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::Conv",
                   "ai.onnx::11::Conv",
                   "ai.onnx::12::Conv",
                   "ai.onnx::13::Conv"})
    .ParseParamsFn(ParseParamsConv2D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
