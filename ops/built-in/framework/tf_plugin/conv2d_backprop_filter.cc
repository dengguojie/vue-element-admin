/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
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
Status ParseParamsConv2DBackpropFilter(const Message* op_src, ge::Operator& op)
{
    OP_LOGI(op.GetName().c_str(), "Enter ParseParamsConv2DBackpropFilter.");

    AutoMappingFn(op_src, op);

    const int32_t CV_NUM_0 = 0;
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorW = op_dsc->GetOutputDesc(CV_NUM_0);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op_dsc->UpdateOutputDesc(CV_NUM_0, orgTensorW);
    if(ret != ge::GRAPH_SUCCESS)
    {
        OP_LOGE(op.GetName().c_str(), "Update filter format failed!");
        map<string, string> err_map;
        err_map["op_name"] = "Conv2dBackpropFilter";
        err_map["param_name"] = "updating output_desc's format";
        err_map["rule_desc"] = "updata output_desc format ";
        err_map["param_value"] = "failed";
        std::string report_error_code = "E50012";
        (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    } else {
        OP_LOGI(op.GetName().c_str(), "Update filter format success, now is %d",op.GetInputDesc(CV_NUM_0).GetFormat());
    }

    // Escape GE require attr [pads] check here
    std::vector<int32_t> padList = {0,0,0,0};
    op.SetAttr("pads", padList);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2DBackpropFilter")
    .ParseParamsFn(ParseParamsConv2DBackpropFilter)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
