/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <map>
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "operator.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"

using namespace ge;

namespace domi {

const int POS_0 = 0;
const int POS_1 = 1;

Status ParseParamsConv3D(const Message* op_src, ge::Operator& op) {

    AutoMappingFn(op_src, op);
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(POS_1);
    orgTensorW.SetOriginFormat(ge::FORMAT_DHWCN);
    orgTensorW.SetFormat(ge::FORMAT_DHWCN);
    auto ret = op_dsc->UpdateInputDesc(POS_1, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update filter format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3d";
        err_map["param_name"] = "updating filter's format";
        err_map["rule_desc"] = "update filter's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "update filter format success, now is %d", op.GetInputDesc(POS_1).GetFormat());

    ge::GeTensorDesc orgTensorX = op_dsc->GetInputDesc(POS_0);
    orgTensorX.SetOriginFormat(ge::FORMAT_NDHWC);
    orgTensorX.SetFormat(ge::FORMAT_NDHWC);
    ret = op_dsc->UpdateInputDesc(POS_0, orgTensorX);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update input x format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3d";
        err_map["param_name"] = "updating input_x's format";
        err_map["rule_desc"] = "update input_x's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }
    OP_LOGI(op.GetName().c_str(), "update input x format success, now is %d", op.GetInputDesc(POS_0).GetFormat());

    ge::GeTensorDesc orgTensorY = op_dsc->GetOutputDesc(POS_0);
    orgTensorY.SetOriginFormat(ge::FORMAT_NDHWC);
    orgTensorY.SetFormat(ge::FORMAT_NDHWC);
    ret = op_dsc->UpdateOutputDesc(POS_0, orgTensorY);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output y format failed.");
        map<string, string> err_map;
        err_map["op_name"] = "Conv3d";
        err_map["param_name"] = "updating output_y's format";
        err_map["rule_desc"] = "update output_y's format";
        err_map["format"] = "failed";
        std::string report_error_code = "E50012";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }
    std::vector<int32_t> padList = {0, 0, 0, 0, 0, 0};
    op.SetAttr("pads", padList);

    OP_LOGI(op.GetName().c_str(), "update output y format success, now is %d", op.GetInputDesc(POS_0).GetFormat());
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3D")
    .ParseParamsFn(ParseParamsConv3D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

