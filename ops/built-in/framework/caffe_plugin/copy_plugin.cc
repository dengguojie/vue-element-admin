/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the
License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

using namespace ge;

namespace domi
{
// Caffe ParseParams
Status ParseParamsCopy(const Message* op_src, ge::Operator& op_dst)
{
    OP_LOGI("Copy", "enter into ParseParamsCopy  ------begin!!");

    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_src);
    if (nullptr == layer)
    {
        OP_LOGE("Copy", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    int64_t n = layer->top_size();
    op_dst.SetAttr("N",n);
    OP_LOGI(op_dst.GetName().c_str(),
        "[PLUGIN_Copy]--------------top_size=%d---------------", n);
    std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
    op_desc->AddDynamicOutputDesc("y", n);

    OP_LOGI("Copy", "ParseParamsCopy ------end!!");

    return SUCCESS;
}

// register Copy operation
REGISTER_CUSTOM_OP("Copy")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("Split")  // name in caffe module
    .ParseParamsFn(ParseParamsCopy)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
