/**
 * Copyright 2020 Huawei Technologies Co., Ltd
*/

#include "register/register.h"
#include "graph/operator.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/error_manager/error_manager.h"
#include <string>
#include <vector>

namespace domi {

// Replace ge ParseParams fuction to process graph conv2d node attrs
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

    // The fmap should be NCHW
    ge::GeTensorDesc orgTensorX = op_dsc->GetInputDesc(0);
    orgTensorX.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorX.SetFormat(ge::FORMAT_NCHW);
    auto retX = op_dsc->UpdateInputDesc(0, orgTensorX);
    if (retX != ge::GRAPH_SUCCESS) {
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "update fmap format failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }

    // The filter should be NCHW
    ge::GeTensorDesc orgTensorW = op_dsc->GetInputDesc(1);
    orgTensorW.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorW.SetFormat(ge::FORMAT_NCHW);
    auto retW = op_dsc->UpdateInputDesc(1, orgTensorW);
    if (retW != ge::GRAPH_SUCCESS) {
        map<string, string> err_map;
        err_map["op_name"] = op.GetName().c_str();
        err_map["description"] = "update filter format failed.";
        std::string report_error_code = "E50058";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return FAILED;
    }

    // The output should be NCHW
    ge::GeTensorDesc orgTensorY = op_dsc->GetOutputDesc(0);
    orgTensorY.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorY.SetFormat(ge::FORMAT_NCHW);
    auto retY = op_dsc->UpdateOutputDesc(0, orgTensorY);
    if (retY != ge::GRAPH_SUCCESS) {
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
    std::vector<int32_t> stridesListDefault = {1, 1, 1, 1};
    op.SetAttr("strides", stridesListDefault);

    // set default dilations
    std::vector<int32_t> dilationsListDefault = {1, 1, 1, 1};
    op.SetAttr("dilations", dilationsListDefault);

    // set default pads
    std::vector<int32_t> padListDefault = {0, 0, 0, 0};
    op.SetAttr("pads", padListDefault);

    // set default groups
    int32_t groupsDefault = 1;
    op.SetAttr("groups", groupsDefault);

    // set default auto_pad
    std::string autoPadDefault = "NOTSET";
    op.SetAttr("auto_pad", autoPadDefault);

    //set data_format, set to NCHW
    op.SetAttr("data_format", ge::FORMAT_NCHW);


    // if attr is set in model, receive them with these var
    std::vector<int32_t> stridesList;
    std::vector<int32_t> dilationsList;
    std::vector<int32_t> padList;

    // update attrs with model value
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS){
            stridesList.push_back(1);
            stridesList.push_back(1);
            stridesList.push_back(attr.ints(0));
            stridesList.push_back(attr.ints(1));
            op.SetAttr("strides", stridesList);
        }
        else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS){
            dilationsList.push_back(1);
            dilationsList.push_back(1);
            dilationsList.push_back(attr.ints(0));
            dilationsList.push_back(attr.ints(1));
            op.SetAttr("dilations", dilationsList);
        }
        else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS){
            // in onnx pads=[top, left, bottomm, right] -> [top, bottom, left, right]
            padList.push_back(attr.ints(0));
            padList.push_back(attr.ints(2));
            padList.push_back(attr.ints(1));
            padList.push_back(attr.ints(3));
            op.SetAttr("pads", padList);
        }
        else if (attr.name() == "group" && attr.type() == ge::onnx::AttributeProto::INTS){
            op.SetAttr("groups", attr.ints(0));
        }
        else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
            op.SetAttr("auto_pad", attr.s());
        }

    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2D")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Conv")
    .ParseParamsFn(ParseParamsConv2D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
