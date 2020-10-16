/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <map>
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"

using namespace ge;

namespace domi {
    
Status ParseParamsConv3DBackpropInput(const Message* op_src, ge::Operator& op)
{
    OP_LOGI(op.GetName().c_str(), "Enter ParseParamsConv3DBackpropInput.");

    AutoMappingFn(op_src, op);

    const int32_t INDEX_0 = 0;
    const int32_t INDEX_1 = 1;
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(INDEX_1);
    orgTensorW.SetOriginFormat(ge::FORMAT_DHWCN);
    orgTensorW.SetFormat(ge::FORMAT_DHWCN);
    auto ret = op_dsc->UpdateInputDesc(INDEX_1, orgTensorW);
    if(ret != ge::GRAPH_SUCCESS)
    {
        OP_LOGE(op.GetName().c_str(), "Update filter format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dBackpropInput";
        err_map["param_name"] = "updating filter's format";
        err_map["rule_desc"] = "update filter's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "Update filter format success");

    ge::GeTensorDesc orgTensorY = op_dsc->GetInputDesc(INDEX_0);
    orgTensorY.SetOriginFormat(ge::FORMAT_NDHWC);
    orgTensorY.SetFormat(ge::FORMAT_NDHWC);
    ret = op_dsc->UpdateInputDesc(INDEX_0, orgTensorY);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update inout out_backprop format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dBackpropInput";
        err_map["param_name"] = "updating out_backprop's format";
        err_map["rule_desc"] = "update out_backprop's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "update inout out_backprop format success.");


    ge::GeTensorDesc orgTensorX = op_dsc->GetOutputDesc(INDEX_0);
    orgTensorX.SetOriginFormat(ge::FORMAT_NDHWC);
    orgTensorX.SetFormat(ge::FORMAT_NDHWC);
    ret = op_dsc->UpdateOutputDesc(INDEX_0, orgTensorX);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output dx format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dBackpropInput";
        err_map["param_name"] = "updating output_dx's format";
        err_map["rule_desc"] = "update output_dx's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }

    std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
    op.SetAttr("pads", padList);

    OP_LOGI(op.GetName().c_str(), "update output dx format success.");

    OP_LOGI(op.GetName().c_str(), "Exit ParseParamsConv3DBackpropInput.");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3DBackpropInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3DBackpropInputV2")
    .ParseParamsFn(ParseParamsConv3DBackpropInput)
    .ImplyType(ImplyType::TVM);
} // namespace domi
