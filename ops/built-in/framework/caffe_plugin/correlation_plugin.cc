/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

/*!
 * \file caffe_correlation_plugin.cc
 * \brief
 */

#include <memory>
#include <string>
#include <vector>
#include "register/register.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/operator.h"
#include "op_log.h"

using namespace ge;

namespace domi {
Status ParseParamsByOperatorCorrelation(const ge::Operator& op_src, ge::Operator& op_dest)
{
    OP_LOGI("Correlation Parse Params for Correlation begin");
    int groups = 1;
    if (op_src.GetAttr("groups", groups) == ge::GRAPH_SUCCESS) {
      op_dest.SetAttr("groups", groups);
    } else {
      op_dest.SetAttr("groups", 1);
    }
    OP_LOGI("Correlation Parse Params for Correlation end");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Correlation")
   .FrameworkType(CAFFE)
   .OriginOpType("Correlation")
   .ParseParamsByOperatorFn(ParseParamsByOperatorCorrelation)
   .ImplyType(ImplyType::TVM);
}  // namespace domi
