/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>
#include <vector>


namespace domi {

static const std::string DATA_FORMAT = "NCHW";

Status ParseParamsMaxPool(const Message *op_src, ge::Operator &op_dest) {
    OP_LOGI("MaxPool", "[PLUGIN_MaxPool]--------------ParseParamsMaxPool  start---------------");
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE("MaxPool", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int64_t ceil_mode = 0;
    std::vector<int> v_ksizes={};
    std::vector<int> v_strides={};
    std::vector<int> v_pads={};
    std::string v_pad = "SAME";
    std::vector<int> DefaultStride={1, 1};
    std::vector<int> DefaultPads={1, 1, 1, 1};

    bool set_ksizes_flag = false;
    bool set_strides_flag = false;
    bool set_pads_flag = false;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == 2) {
                for (int i = 0; i < attr.ints_size(); i++) {
                    v_ksizes.push_back(attr.ints(i));
                }
            }
            else if (attr.ints_size() == 1) {
                v_ksizes.push_back(attr.ints(0));
                v_ksizes.push_back(attr.ints(0));
            }
            set_ksizes_flag = true;
        }

        else if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == 2) {
                for (int i = 0; i < attr.ints_size(); i++) {
                    v_strides.push_back(attr.ints(i));
                }
            }
            else if (attr.ints_size() == 1) {
                v_strides.push_back(attr.ints(0));
                v_strides.push_back(attr.ints(0));
            }
            set_strides_flag = true;
        }

        else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
            if (attr.s() == "VALID") {
                v_pad = "VALID";
            }
            else {
                v_pad = "SAME";
            }
        }

        else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == 4) {
                for (int i = 0; i < attr.ints_size(); i++) {
                    v_pads.push_back(attr.ints(i));
                }
            }
            else if (attr.ints_size() == 1) {
                v_pads.push_back(attr.ints(0));
                v_pads.push_back(attr.ints(0));
                v_pads.push_back(attr.ints(0));
                v_pads.push_back(attr.ints(0));
            }
            set_pads_flag = true;
        }

        else if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
            ceil_mode = attr.i();
        }
    }

    if (ceil_mode == 0) {
        op_dest.SetAttr("ceil_mode", 1);
    } else {
        op_dest.SetAttr("ceil_mode", 0);
    }

    if (set_ksizes_flag) {
        op_dest.SetAttr("window", v_ksizes);
    }
    else {
        OP_LOGI("MaxPool", "onnx MaxPool op has no ksize attr");
        op_dest.SetAttr("window", DefaultStride);
    }

    if (set_strides_flag) {
        op_dest.SetAttr("stride", v_strides);
    }
    else {
        OP_LOGI("MaxPool", "onnx MaxPool op has no strides attr, use default.");
        op_dest.SetAttr("strides", DefaultStride);
    }

    if (set_pads_flag) {
        op_dest.SetAttr("pad", v_pads);
    }
    else {
        OP_LOGI("MaxPool", "onnx MaxPool op has no pads attr, use default.");
        op_dest.SetAttr("pad", DefaultPads);
    }
     
    OP_LOGI("MaxPool", "--------------ParseParamsMaxPool  end---------------");

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Pooling")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::MaxPool")
    .ParseParamsFn(ParseParamsMaxPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
