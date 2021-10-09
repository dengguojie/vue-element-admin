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
 * \file nll_loss_grad.h
 * \brief
 */
#ifndef __NLL_LOSS_GRAD_H__
#define __NLL_LOSS_GRAD_H__

#include <string>

namespace optiling {

struct NLLLossGradTilingParam {
  int64_t c_dim = 0;
  int64_t n_dim = 0;
  int64_t invalid_target = 0;
  int64_t ignore_idx = 0;
  int64_t output_gm_size = 0;
  int64_t x_gm_size = 0;
  int64_t y_grad_gm_size = 0;
  int64_t target_gm_size = 0;
  int64_t data_total_weight_size = 0;
  int64_t weight_gm_size = 0;
  int64_t big_weight = 0;
  int64_t core_num = 0;
  int64_t max_line = 0;
  int64_t lower_line = 0;
  int64_t loop_time = 0;
  int64_t fake_core = 0;
  int64_t redundant_line = 0;
  int64_t max_total_num = 1;
  int64_t lower_total_num = 0;
  int64_t dup_ub_size = 0;
  int64_t target_ub_size = 0;
  int64_t weight_ub_size = 0;
  int64_t total_weight_ub_size = 0;
  int64_t refactor_weight_ub_size = 0;
  int64_t weight_burst = 0;
  int64_t target_burst = 0;
  int64_t lower_target_burst = 0;
  int64_t max_vmul_repeat = 0;
  int64_t lower_vmul_repeat = 0;
  int64_t last_target_burst = 0;
  int64_t last_vmul_repeat = 0;
  int64_t core_dup_repeat = 0;
  int64_t last_dup_repeat = 0;
  int64_t max_out_burst = 0;
  int64_t last_out_burst = 0;
  int64_t y_grad_ub_size = 0;
  int64_t tiling_key = 0;
  int64_t align_repeat_size = 0;
  int64_t move_out_time = 0;
  int64_t single_max_repeat = 0;
  int64_t tail_repeat = 0;
  int64_t offet = 0;
};

/*
 * @brief: get tiling info
 * @param [in] max_move_line: the max line in ub
 * @param [in] reduction: 'none'|'mean'|'sum'
 * @param [out] tiling_param: get the tiling info
 */
bool GetTilingParamOfNormalTwoDim(const int64_t max_move_line, const std::string& reduction,
                                  NLLLossGradTilingParam& tiling_param);
}  // namespace optiling

#endif  // __NLL_LOSS_GRAD_H__
