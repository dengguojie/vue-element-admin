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
 * \file transpose.h
 * \brief
 */
#ifndef __AS_STRIDED_H__
#define __AS_STRIDED_H__

#include <vector>
#include <string>
#include <map>
#include <queue>
#include <memory>
#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "register/op_tiling.h"

namespace optiling {

struct AsStridedInfo {
    int64_t axisLen;
    int64_t stride;
    AsStridedInfo() {
        axisLen = 0;
        stride = 0;
    }
};

}// namespace optiling

#endif  //__AS_STRIDED_H__
