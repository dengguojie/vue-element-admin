/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>
#include <vector>
#include <map>
#include "common/util/error_manager/error_manager.h"

namespace domi {

Status ParseParamsTfIdfVectorizer(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    map<string, string> errMap;
    errMap["op_name"] = "TfIdfVectorizer";
    errMap["description"] = "Dynamic cast op_src to NodeProto failed!";
    std::string reportErrorCode = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(reportErrorCode, errMap);
    return FAILED;
  }

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "max_gram_length" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t max_gram_length = attr.i();
      op_dest.SetAttr("max_gram_length", max_gram_length);
    }
    if (attr.name() == "max_skip_count" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t max_skip_count = attr.i();
      op_dest.SetAttr("max_skip_count", max_skip_count);
    }
    if (attr.name() == "min_gram_length" && attr.type() == ge::onnx::AttributeProto::INT) {
      int64_t min_gram_length = attr.i();
      op_dest.SetAttr("min_gram_length", min_gram_length);
    }
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      std::string mode = attr.s();
      op_dest.SetAttr("mode", mode);
    }
    if (attr.name() == "ngram_counts" && attr.type() == ge::onnx::AttributeProto::INTS) {
      std::vector<int64_t> ngram_counts;
      for (auto s : attr.ints()) {
        ngram_counts.push_back(s);
      }
      op_dest.SetAttr("ngram_counts", ngram_counts);
    }
    if (attr.name() == "ngram_indexes" && attr.type() == ge::onnx::AttributeProto::INTS) {
      std::vector<int64_t> ngram_indexes;
      for (auto s : attr.ints()) {
        ngram_indexes.push_back(s);
      }
      op_dest.SetAttr("ngram_indexes", ngram_indexes);
    }
    if (attr.name() == "pool_int64s" && attr.type() == ge::onnx::AttributeProto::INTS) {
      std::vector<int64_t> pool_int64s;
      for (auto s : attr.ints()) {
        pool_int64s.push_back(s);
      }
      op_dest.SetAttr("pool_int64s", pool_int64s);
    }
    if (attr.name() == "pool_strings" && attr.type() == ge::onnx::AttributeProto::STRINGS) {
      std::vector<std::string> pool_strings;
      for (auto s : attr.strings()) {
        pool_strings.push_back(s);
      }
      op_dest.SetAttr("pool_strings", pool_strings);
    }
    if (attr.name() == "weights" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
      std::vector<float> weights;
      for (auto s : attr.floats()) {
        weights.push_back(s);
      }
      op_dest.SetAttr("weights", weights);
    }
  }

  return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("TfIdfVectorizer")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::TfIdfVectorizer", "ai.onnx::10::TfIdfVectorizer", "ai.onnx::11::TfIdfVectorizer",
                   "ai.onnx::12::TfIdfVectorizer", "ai.onnx::13::TfIdfVectorizer"})
    .ParseParamsFn(ParseParamsTfIdfVectorizer)
    .ImplyType(ImplyType::TVM);
}  // namespace domi