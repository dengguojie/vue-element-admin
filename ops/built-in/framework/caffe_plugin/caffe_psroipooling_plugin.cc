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

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"


using namespace ge;

namespace domi
{
// Caffe ParseParams
Status ParseParams_PSROIPooling(const Message* op_origin, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    const caffe::LayerParameter* layer =
        dynamic_cast<const caffe::LayerParameter*>(op_origin);

    if (nullptr == layer) {
        OP_LOGE("PSROIPooling", "Dynamic cast op_src to LayerParameter failed.");
        return FAILED;
    }

    const caffe::PSROIPoolingParameter& param = layer->psroi_pooling_param();
    if(param.has_output_dim()) {
        op_dest.SetAttr("output_dim", (int)param.output_dim());
    }
    if(param.has_group_size()) {
        op_dest.SetAttr("group_size", (int)param.group_size());
    }
    if(param.has_spatial_scale()) {
        op_dest.SetAttr("spatial_scale", (float)param.spatial_scale());
    }

    return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("PSROIPooling")
    .FrameworkType(CAFFE)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("PSROIPooling")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_PSROIPooling)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);

}  // namespace domi
