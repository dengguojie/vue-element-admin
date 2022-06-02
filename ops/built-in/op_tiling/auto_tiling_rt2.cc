/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file auto_tiling_rt2.cpp
 * \brief Auto-Tiling Private Declarations and Implementations
 */
#include "auto_tiling_rt2.h"
#include <mutex>

#include "exe_graph/runtime/kernel_run_context.h"
#include "register/op_impl_registry.h"

#include "vector_tiling_log.h"
#include "error_log.h"
#include "auto_tiling_register.h"
#include "vector_tiling_rt2.h"

namespace optiling {
std::string GetStrPattern(const SchPattern pattern) {
  const std::unordered_map<SchPattern, std::string> pattern_maps {
    {SchPattern::ELETWISE, "ElemWise"},
    {SchPattern::BROADCAST, "Broadcast"},
    {SchPattern::COMMONREDUCE, "CommReduce"},
    {SchPattern::TUPLEREDUCE, "TupleReduce"},
    {SchPattern::NORM, "Norm"},
    {SchPattern::CONCAT, "Concat"},
    {SchPattern::SPLIT, "Split"},
    {SchPattern::SLICE, "Slice"},
    {SchPattern::GATHER, "Gather"},
    {SchPattern::TRANSPOSE, "Transpose"},
    {SchPattern::TRANSDATA, "Transdata"},
  };
  if (pattern_maps.find(pattern) == pattern_maps.end()) {
    return "default";
  }
  return pattern_maps.at(pattern);
}

SchPattern GetSchPattern(const std::string& s_pattern) {
  const std::unordered_map<std::string, SchPattern> pattern_maps {
    {"ElemWise", SchPattern::ELETWISE},
    {"Broadcast", SchPattern::BROADCAST},
    {"CommReduce", SchPattern::COMMONREDUCE},
    {"TupleReduce", SchPattern::TUPLEREDUCE},
    {"Norm", SchPattern::NORM},
    {"Concat", SchPattern::CONCAT},
    {"Split", SchPattern::SPLIT},
    {"Slice", SchPattern::SLICE},
    {"Gather", SchPattern::GATHER},
    {"Transpose", SchPattern::TRANSPOSE},
    {"Transdata", SchPattern::TRANSDATA},
  };
  if (pattern_maps.find(s_pattern) == pattern_maps.end()) {
    return SchPattern::DEFAULT;
  }
  return pattern_maps.at(s_pattern);
}

bool AutoTilingRun(SchPattern pattern, gert::TilingContext* context, const OpInfoImpl* op_info) {
  auto finder = AutoTilingRegister::RegisterTiling().find(pattern);
  if (finder == AutoTilingRegister::RegisterTiling().end()) {
    OP_LOGE(context->GetNodeType(), "Pattern %s is not supported by AutoTiling",
            GetStrPattern(pattern).c_str());
    return false;
  }
  if (!finder->second(context, op_info)) {
    OP_LOGE(context->GetNodeType(), "Autotiling func failed");
    return false;
  }
  return true;
}

AutoTilingCompileInfo* ParseAutoTilingRun(const char* op_type, const nlohmann::json& json_compile_info) {
  // Default parsing sequence, parse compile info directly to nlohmann::json object
  OP_TILING_CHECK((json_compile_info.find("_pattern") == json_compile_info.end()),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [_pattern]"),
                  return nullptr;);
  OP_TILING_CHECK(!json_compile_info["_pattern"].is_string(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info[_pattern] not is string"),
                  return nullptr;);
  const string& s_pattern = json_compile_info["_pattern"];
  SchPattern pattern = GetSchPattern(s_pattern);
  auto finder = AutoTilingRegister::RegisterParser().find(pattern);
  if (finder == AutoTilingRegister::RegisterParser().end()) {
    OP_LOGE(op_type, "Pattern %s is not supported by AutoTiling Compile Info Parser",
            s_pattern.c_str());
    return nullptr;
  }
  AutoTilingCompileInfo* compile_info = finder->second(op_type, json_compile_info);
  if (compile_info == nullptr) {
    OP_LOGE(op_type, "Failed to parse %s pattern compilation information",
            s_pattern.c_str());
    return nullptr;
  }
  OP_LOGD(op_type, "Auto tiling parser save pattern is %s.", GetStrPattern(pattern).c_str());
  OP_LOGD(op_type, "Auto tiling parser save compile_info is %p.", compile_info);
  compile_info->pattern = pattern;
  return compile_info;
}

/*
 * @brief: tiling function of custom ops
 * @param [in] context: inputs/outputs/attrs of ops
 * @param [in] op_info: inputs/in_type/attrs/compile_info of ops
 * @return bool: success or not
 */
bool DoAutoTiling(gert::TilingContext* context, const OpInfo* op_info) {
  OP_LOGI(context->GetNodeType(), "Entering customized autoTiling.");
  auto op_info_impl = OpInfoImplGetter::GetOpInfoImpl(op_info);
  if (op_info == nullptr || op_info_impl == nullptr || op_info_impl->GetCompileInfo() == nullptr) {
    OP_LOGE(context->GetNodeType(), "Op info cannot be empty");
    return false;
  }
  auto pattern = op_info_impl->GetCompileInfo()->pattern;
  OP_LOGD(context->GetNodeType(), "Custom auto tiling get pattern is %s.", GetStrPattern(pattern).c_str());
  return AutoTilingRun(pattern, context, op_info_impl.get());
}

/*
 * @brief: tiling parser function of custom ops
 * @param [in] op_type: op_type of ops
 * @param [in] json_compile_info: compile_info of ops
 * @return std::shared_ptr<AutoTilingCompileInfo>: AutoTilingCompileInfo pointer
 */
std::shared_ptr<AutoTilingCompileInfo> ParseAutoTiling(
    const char* op_type, const nlohmann::json& json_compile_info) {
  return std::shared_ptr<AutoTilingCompileInfo>(ParseAutoTilingRun(op_type, json_compile_info));
}

/*
 * @brief: tiling function of ops
 * @param [in] context: inputs/outputs/attrs of ops
 * @return bool: success or not
 */
ge::graphStatus AutoTiling(gert::TilingContext *context) {
  OP_LOGI("AutoTiling", "Entering AutoTiling.");
  auto* compile_info = reinterpret_cast<const AutoTilingCompileInfo*>(context->GetCompileInfo());
  if (compile_info == nullptr) {
    OP_LOGE(context->GetNodeType(), "Autotiling compile info is null");
    return ge::GRAPH_FAILED;
  }
  OP_LOGD(context->GetNodeType(), "Auto tiling get compile_info is %p.", compile_info);
  OP_LOGD(context->GetNodeType(), "Auto tiling get pattern is %s.", GetStrPattern(compile_info->pattern).c_str());
  if (!AutoTilingRun(compile_info->pattern, context, nullptr)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

/*
 * @brief: tiling parser function of ops
 * @param [in] context: compile info of ops
 * @return bool: success or not
 */
ge::graphStatus AutoTilingParser(gert::KernelContext *context) {
  OP_LOGI("AutoTiling", "Entering AutoTiling parser.");
  auto json_str = context->GetInputStrPointer(0);
  const char* op_type = reinterpret_cast<const gert::ComputeNodeInfo*>(context->GetComputeNodeExtend())->GetNodeType();
  OP_LOGD(op_type, "Receive compile info is %s.", json_str);
  if (json_str == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const nlohmann::json& json_compile_info = nlohmann::json::parse(json_str);
  AutoTilingCompileInfo* compile_info = ParseAutoTilingRun(op_type, json_compile_info);
  if (compile_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto out = context->GetOutput(0);
  auto deleter = [](void* obj) {
    delete static_cast<AutoTilingCompileInfo*>(obj);
  };
  out->Set(compile_info, deleter);
  return ge::GRAPH_SUCCESS;
}

// register AutoTiling for runtime 2.0
IMPL_OP_DEFAULT().Tiling(AutoTiling).TilingParse<AutoTilingCompileInfo>(AutoTilingParser);
}  // namespace optiling
