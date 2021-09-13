/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file resize_common.h
 * \brief
 */
#include <string>
#include <math.h>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

// tiling_key format: 000000
// 1. Reserved, default 1
// 2. h align flag, 0: h -> x.x*h, 1: h -> nh, 2: nh -> h, 3: h = h
// 3. w align flag, 0: w -> x.x*w, 1: w -> nw, 2: nw -> w, 3: w = w
// 4. Reserved, default 0
// 5. Reserved, default 0
// 6. Reserved, default 0
const int64_t DEFAULT_TILING_MODE = 100000;
const int64_t HEIGHT_ALIGN_FLAG = 10000;
const int64_t width_ALIGN_FLAG = 1000;
const int64_t WIDTH_ALIGN_FLAG = 100;
const int64_t BIG_TO_SMALL_FLAG = 10;

// auto tune interface parameter name
const char INNERTUNEPARAM[] = "_tune_param";
const char TUNEPARAM[] = "tune_param";

struct ResizeClassTilingParams {
  int64_t tiling_key;
  int64_t input_batch;
  int64_t input_c1;
  int64_t input_height;
  int64_t input_width;
  int64_t output_height;
  int64_t output_width;
  // cut core num by batch * C1
  int64_t cut_batch_c1_num;
  // cut core num by height
  int64_t cut_height_num;
  // cut core num by width
  int64_t cut_width_num;
};

struct TuneParams {
  int64_t tiling_key = 0;
  int64_t cut_batch_c1_num = 0;
  int64_t cut_height_num = 0;
  int64_t cut_width_num = 0;
};

struct ResizeClassCompileParams {
  int64_t core_num;
  int64_t max_w_len;
  int64_t align_corners;
  int64_t half_pixel_centers;
  TuneParams tuneParams;
  std::string op_type;
};

/*
 * @brief: print the tiling info and compile info
 * @param [in] compile_info: the compile json info
 * @param [in] compile_params: the compile info struct
 * @param [out] compile_params: set the compile_params from compile_info json
 */
bool GetResizeClassCompileParams(const nlohmann::json& compile_info, ResizeClassCompileParams& compile_params);

/*
 * @brief: set the tuneParams of compile_params
 * @param [in] compile_info: the compile json info
 * @param [in] compile_params: the compile info struct
 * @param [out] compile_params: set the tuneParams from compile_info json
 */
bool GetResizeClassTuneParams(const nlohmann::json& compile_info, ResizeClassCompileParams& compile_params);

/*
 * @brief: print the tiling info and compile info
 * @param [in] op_type: the op name
 * @param [in] tiling_params: the tiling info struct
 * @param [in] compile_params: the compile info struct
 */
void PrintTilingParams(const std::string& op_type, const ResizeClassTilingParams& tiling_params,
                       const ResizeClassCompileParams& compile_params);

/*
 * @brief: set the tiling info to OpRunInfo
 * @param [in] tiling_params: the tiling info struct
 * @param [in] run_info: OpRunInfo
 * @param [out] run_info: set the tiling info to OpRunInfo
 */
void SetTilingParams(const ResizeClassTilingParams& tiling_params, OpRunInfo& run_info);

/*
 * @brief: tiling function for HW2MHNW
 * @param [in] compile_params: the compile info struct
 * @param [in] tiling_params: the tiling info struct
 * @param [out] tiling_params: modify the tiling var in struct tiling_params
 */
void GetTilingForHW2MHNW(const ResizeClassCompileParams& compile_params, ResizeClassTilingParams& tiling_params);

/*
 * @brief: tiling function of ResizeNearestNeighborV2
 * @param [in] compile_params: the compile info struct
 * @param [in] tiling_params: the tiling info struct
 * @param [out] tiling_params: modify the tiling var in struct tiling_params
 * @return bool: success or not
 */
bool GetResizeNearestNeighborV2Tiling(const ResizeClassCompileParams& compile_params,
                                      ResizeClassTilingParams& tiling_params);

/*
 * @brief: tiling function of ResizeNearestNeighborV2Grad
 * @param [in] compile_params: the compile info struct
 * @param [in] tiling_params: the tiling info struct
 * @param [out] tiling_params: modify the tiling var in struct tiling_params
 * @return bool: success or not
 */
bool GetResizeNearestNeighborV2GradTiling(const ResizeClassCompileParams& compile_params,
                                          ResizeClassTilingParams& tiling_params);

/*
 * @brief: tiling function of ResizeBilinearV2
 * @param [in] compile_params: the compile info struct
 * @param [in] tiling_params: the tiling info struct
 * @param [out] tiling_params: modify the tiling var in struct tiling_params
 * @return bool: success or not
 */
bool GetResizeBilinearV2Tiling(const ResizeClassCompileParams& compile_params, ResizeClassTilingParams& tiling_params);
}  // namespace optiling
