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

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "operator.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"

using namespace ge;
namespace domi {

const int kInputFilter = 1;

// Replace ge ParseParams fuction to process graph conv2d node attrs
Status ParseParamsDeformableConv2D(const Message* op_src, ge::Operator& op) {

    // Convert original tf graph conv2d attrs to GE graph attrs
    AutoMappingFn(op_src, op);

    // The filter format shuold be HWCN, not NHWC or NCHW, so set here to fix this problem
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc org_tensor_w = op_dsc->GetInputDesc(kInputFilter);
    org_tensor_w.SetOriginFormat(ge::FORMAT_HWCN);
    org_tensor_w.SetFormat(ge::FORMAT_HWCN);
    auto ret = op_dsc->UpdateInputDesc(kInputFilter, org_tensor_w);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update filter format failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "update filter format failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DeformableConv2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"DeformableConv2D", "DeformableConv2DWithBias"})
    .ParseParamsFn(ParseParamsDeformableConv2D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
