/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: warpPerspective_plugin.cpp c3x pasrser cpp file
 * Author: huawei
 * Create: 2020-6-11
 * Note:
 */

#include <memory>
#include <string>
#include <vector>
#include "register/register.h"

using namespace ge;
namespace domi {

Status ParseParamsImagePad(const ge::Operator& op_src, ge::Operator& op)
{
    int padValue = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("pad_value", padValue)) {
        op.SetAttr("pad_value", padValue);
    }
    const int padAttrSize = 6;
    vector<int> dimSize;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("dim", dimSize)) {
        if (dimSize.size() != padAttrSize) {
            printf("[ERROR][Plugin] ImagePad's dim only support 6!\n");
            return FAILED;
        }
    }

    const int padRows = 3;
    const int padCols = 2;
    int index = 0;
    vector<vector<int64_t>> paddings(padRows, vector<int64_t>(padCols));
    for (int i = 0; i < padRows; i++) {
        for (int j = 0; j < padCols; j++) {
            paddings[i][j] = dimSize[index];
            index++;
        }
    }

    op.SetAttr("paddings", paddings);
    TensorDesc opDesc = op.GetInputDesc(0);
    opDesc.SetOriginFormat(ge::FORMAT_ND);
    opDesc.SetFormat(ge::FORMAT_ND);
    auto ret = op.UpdateInputDesc("input_dict", opDesc);
    if (ret != ge::GRAPH_SUCCESS) {
         return FAILED;
    }
 
    printf("[INFO][Plugin]--------------ParseParams ImagePad End---------------\n");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("ImagePad")
    .FrameworkType(CAFFE)
    .OriginOpType("ImagePad")
    .ParseParamsByOperatorFn(ParseParamsImagePad)
    .ImplyType(ImplyType::TVM);
}
