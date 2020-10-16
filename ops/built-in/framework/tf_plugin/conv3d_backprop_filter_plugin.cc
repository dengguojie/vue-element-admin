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

Status ParseParamsConv3DBackpropFilter(const Message* op_src, ge::Operator& op)
{
    OP_LOGI(op.GetName().c_str(), "Enter ParseParamsConv3DBackpropFilter.");
    AutoMappingFn(op_src, op);

    const int32_t inputIdx0 = 0;
    const int32_t inputIdx1 = 1;
    const int32_t outputIdx0 = 0;
    auto opDsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorX = opDsc->GetInputDesc(inputIdx0);
    orgTensorX.SetOriginFormat(ge::FORMAT_NDHWC);
    orgTensorX.SetFormat(ge::FORMAT_NDHWC);
    auto ret = opDsc->UpdateInputDesc(inputIdx0, orgTensorX);
    if(ret != ge::GRAPH_SUCCESS)
    {
        OP_LOGE(op.GetName().c_str(), "Update input_x format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dBackpropFilter";
        err_map["param_name"] = "updating input_x's format";
        err_map["rule_desc"] = "update input_x's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }

    ge::GeTensorDesc orgTensorY = opDsc->GetInputDesc(inputIdx1);
    orgTensorY.SetOriginFormat(ge::FORMAT_NDHWC);
    orgTensorY.SetFormat(ge::FORMAT_NDHWC);
    ret = opDsc->UpdateInputDesc(inputIdx1, orgTensorY);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Update out_backprop format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dBackpropFilter";
        err_map["param_name"] = "updating out_backprop's format";
        err_map["rule_desc"] = "update out_backprop's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }

    ge::GeTensorDesc orgTensorW = opDsc->GetOutputDesc(outputIdx0);
    orgTensorW.SetOriginFormat(ge::FORMAT_DHWCN);
    orgTensorW.SetFormat(ge::FORMAT_DHWCN);
    ret = opDsc->UpdateOutputDesc(inputIdx0, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Update output dw format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3dBackpropFilter";
        err_map["param_name"] = "updating output_dw's format";
        err_map["rule_desc"] = "update output_dw's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }
    std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
    op.SetAttr("pads", padList);

    OP_LOGI(op.GetName().c_str(), "update output dw format success.");

    OP_LOGI(op.GetName().c_str(), "Exit ParseParamsConv3DBackpropFilter.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3DBackpropFilterV2")
    .ParseParamsFn(ParseParamsConv3DBackpropFilter)
    .ImplyType(ImplyType::TVM);
} // namespace domi
