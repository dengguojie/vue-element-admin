/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 * Description: Tile framework plugin cpp file
 * Author:
 * Create: 2020-6-17
 * Note:
 */

#include <memory>
#include <string>
#include <vector>
#include "register/register.h"

namespace domi {

    Status ParseParamsTile(const ge::Operator &op_src, ge::Operator &op_dst)
    {
        int axis = 0;
        int tiles = 0;
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("axis", axis)) {
            op_dst.SetAttr("axis", axis);
        }
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("tiles", tiles)) {
            op_dst.SetAttr("tiles", tiles);
        }

        return SUCCESS;
    }

    REGISTER_CUSTOM_OP("TileCaffe")
    .FrameworkType(CAFFE)
    .OriginOpType("Tile")
    .ParseParamsByOperatorFn(ParseParamsTile)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

