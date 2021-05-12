/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Plugin for Operator Heatmap2Coord
 * Author: Huawei
 * Create: 2020-06-12
 */

#include <memory>
#include <string>
#include <vector>
#include "register/register.h"

namespace domi {
    Status ParseParamsHeatmap2Coord(const ge::Operator& op_src, ge::Operator &op_dest)
    {
        return SUCCESS;
    }

    // register Heatmap2coord op info to GE
    REGISTER_CUSTOM_OP("Heatmap2Coord")
    .FrameworkType(CAFFE)
    .OriginOpType("Heatmap2Coord")
    .ParseParamsByOperatorFn(ParseParamsHeatmap2Coord)
    .ImplyType(ImplyType::TVM);
}

