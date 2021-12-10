/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file tiling_handler.h
 * \brief
 */
#ifndef __TILINGHANDLER_H__
#define __TILINGHANDLER_H__

#include "vector_tiling.h"

namespace optiling {
std::shared_ptr<AutoTilingHandler> CreateNormTilingHandler(const std::string& op_type,
                                                           const std::string& pattern,
                                                           const nlohmann::json& parsed_compile_info);

std::shared_ptr<AutoTilingHandler> CreateTransdataTilingHandler(const std::string& op_type,
                                                                const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info);

std::shared_ptr<AutoTilingHandler> CreateTransposeDslTilingHandler(const std::string& op_type,
                                                                   const std::string& pattern,
                                                                   const nlohmann::json& parsed_compile_info);

std::shared_ptr<AutoTilingHandler> CreateConcatDslTilingHandler(const std::string& op_type,
                                                                const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info);

std::shared_ptr<AutoTilingHandler> CreateElewiseTilingHandler(const std::string& op_type,
                                                              const std::string& pattern,
                                                              const nlohmann::json& parsed_compile_info);

std::shared_ptr<AutoTilingHandler> CreateBroadcastTilingHandler(const std::string& op_type,
                                                                const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info);

std::shared_ptr<AutoTilingHandler> CreateReduceTilingHandler(const std::string& op_type,
                                                             const std::string& pattern,
                                                             const nlohmann::json& parsed_compile_info);

std::shared_ptr<AutoTilingHandler> CreateGatherTilingHandler(const std::string& op_type,
                                                             const std::string& pattern,
                                                             const nlohmann::json& parsed_compile_info);
}  // namespace optiling
#endif  //__TILINGHANDLER_H__
