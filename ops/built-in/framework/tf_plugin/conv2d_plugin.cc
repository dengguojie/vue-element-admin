/**
 * Copyright 2020 Huawei Technologies Co., Ltd
*/

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "operator.h"
#include "op_log.h"
#include "common/util/error_manager/error_manager.h"

using namespace ge;
namespace domi {

const int INPUT_FILTER = 1;

// Replace ge ParseParams fuction to process graph conv2d node attrs
Status ParseParamsConv2D(const Message* op_src, ge::Operator& op) {

    // Convert original tf graph conv2d attrs to GE graph attrs
    AutoMappingFn(op_src, op);

    // The filter format shuold be HWCN, not NHWC or NCHW, so set here to fix this problem
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(INPUT_FILTER);
    orgTensorW.SetOriginFormat(ge::FORMAT_HWCN);
    orgTensorW.SetFormat(ge::FORMAT_HWCN);
    auto ret = op_dsc->UpdateInputDesc(INPUT_FILTER, orgTensorW);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update filter format failed.");
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "update filter format failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }

    // Escape GE require attr [pads] check here
    std::vector<int32_t> padList = {0,0,0,0};
    op.SetAttr("pads", padList);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2D")
    .ParseParamsFn(ParseParamsConv2D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

