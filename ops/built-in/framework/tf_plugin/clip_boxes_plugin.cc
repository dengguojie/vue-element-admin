/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file clip_boxes_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {
Status ClipBoxesParams(const std::vector<const google::protobuf::Message*> insideNodes, ge::Operator& op) {
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ClipBoxes")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ClipBoxes")
    .FusionParseParamsFn(ClipBoxesParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
