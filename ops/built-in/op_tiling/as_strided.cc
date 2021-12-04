/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file as_strided.cc
 * \brief tiling function of op as_strided
 */

#include <string>
#include <vector>

#include "graph/debug/ge_log.h"
#include "error_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"

#include "as_strided.h"

using namespace std;

using namespace ge;
namespace optiling {


// define the compile key of json.vars
static const std::vector<std::string> COMPILE_INFO_KEY = {"max_elem_cnt", "core_num"};

static int64_t GetFloorDiv(const int64_t u_value, const int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = u_value / d_value;

  return res_value;
}

static int64_t GetCeilDiv(const int64_t u_value, const int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = (u_value + d_value - 1) / d_value;

  return res_value;
}

static int64_t GetDivisorAlign(const int64_t u_value, const int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = u_value / d_value * d_value;

  return res_value;
}

static int64_t GetRangeSize(const std::vector<int64_t>& in_shape, const int64_t beg, int64_t end) {
  int64_t range_size = 1;

  if (end < 0) {
    end = beg;
  }

  for (int64_t i = beg; i <= end; i++) {
    range_size *= in_shape[i];
  }

  return range_size;
}

static int64_t GetElemIndexInOri(const AsStridedInfo& as_info, const int64_t row, const int64_t col) {
  int64_t elem_index_in_ori = 0;

  // row and col are 1-based
  if (row == 0 || col == 0) {
    elem_index_in_ori = 1;
  } else {
    int64_t n_row = row - 1;
    for (int64_t i = 0; i < as_info.dim_num; i++) {
      elem_index_in_ori += (n_row / as_info.dim_except_last_paras[i*3+0] %
                            as_info.dim_except_last_paras[i*3+1] *
                            as_info.dim_except_last_paras[i*3+2]);
    }
    elem_index_in_ori = elem_index_in_ori + as_info.storage_offset + 1 + (col - 1) * as_info.last_dim_stride;
  }

  return elem_index_in_ori;
}

static bool MergeAxis(const std::vector<int64_t>& out_size, const std::vector<int64_t>& out_stride,
                      std::vector<int64_t>& new_size, std::vector<int64_t>& new_stride) {
  int64_t idx_beg = 0;
  int64_t idx_end = 0;
  int64_t dims = out_stride.size();
  vector<int64_t> tmp_out_size;
  vector<int64_t> tmp_out_stride;

  if (dims == 1) {
    new_stride.push_back(out_stride[0]);
    new_size.push_back(out_size[0]);

    return true;
  }

  // delete axis which size is 1
  for (int64_t i = 0; i < dims; i++) {
    if (out_size[i] != 1) {
      tmp_out_size.push_back(out_size[i]);
      tmp_out_stride.push_back(out_stride[i]);
    }
  }

  int64_t tmp_dims = tmp_out_size.size();
  if (tmp_dims == 1) {
    new_size.push_back(tmp_out_size[0]);
    new_stride.push_back(tmp_out_stride[0]);

    return true;
  }

  if (tmp_dims == 0) {
    new_size.push_back(1);
    new_stride.push_back(0);

    return true;
  }

  /***
   *** stride[0, 0, 1, 0, 0] -> stride[0, 1, 0]
   *** size[1, 2, 3, 4, 5] -> size[2, 3, 20]
  ***/
  new_stride.push_back(tmp_out_stride[0]);
  for (int64_t i = 0; i < tmp_dims - 1; i++) {
    if (tmp_out_stride[i] == tmp_out_stride[i+1] && tmp_out_stride[i] == 0) {
      idx_end = i + 1;
    } else {
      if (idx_end > idx_beg) {
        new_size.push_back(GetRangeSize(tmp_out_size, idx_beg, idx_end));
      } else {
        new_size.push_back(tmp_out_size[idx_beg]);
      }
      idx_beg = i + 1;
      idx_end = i + 1;
      new_stride.push_back(tmp_out_stride[idx_beg]);
    }
  }
  if (tmp_out_stride[tmp_dims-2] == tmp_out_stride[tmp_dims-1]) {
    new_size.push_back(GetRangeSize(tmp_out_size, idx_beg, idx_end));
  } else {
    new_size.push_back(tmp_out_size[idx_beg]);
  }

  /***
   *** stride[2, 18, 6, 3] -> stride[2, 3]
   *** size[3, 4, 3, 2] -> size[3, 24]
  ***/
  int64_t new_dims = new_size.size();
  if (new_dims == 1) {
    return true;
  } else {
    int64_t tmp_axis_merge_size = 1;
    int64_t tmp_axis_merge_stride = 1;
    int64_t last_dim_index = new_size[new_dims-1] * new_stride[new_dims-1];

    if (last_dim_index != new_stride[new_dims-2]) {
      return true;
    } else if (last_dim_index == new_stride[new_dims-2] && new_dims == 2) {
      tmp_axis_merge_size = GetRangeSize(new_size, 0, 1);
      tmp_axis_merge_stride = new_stride[1];
      new_size.resize(1);
      new_stride.resize(1);
      new_size[0] = tmp_axis_merge_size;
      new_stride[0] = tmp_axis_merge_stride;
    } else {
      int64_t cur_idx = new_dims - 2;
      for (int64_t j = cur_idx; j > 0; j--) {
        if (new_size[j] * last_dim_index == new_stride[j-1]) {
          last_dim_index *= new_size[j];
          cur_idx -= 1;
        }
      }

      tmp_axis_merge_size = GetRangeSize(new_size, cur_idx, new_dims-1);
      tmp_axis_merge_stride = new_stride[new_dims-1];
      new_size.resize(cur_idx + 1);
      new_stride.resize(cur_idx + 1);
      new_size[cur_idx] = tmp_axis_merge_size;
      new_stride[cur_idx] = tmp_axis_merge_stride;
   }

  }

  return true;
}

static bool SetDimsTilingParas(const string& op_type, AsStridedInfo& as_info,
                               const std::vector<int64_t>& out_size, const std::vector<int64_t>& out_stride) {
  if (out_size.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the dimension count cannot be zero!");
    return false;
  }

  auto dims = out_size.size();
  as_info.dim_num = dims - 1;  // except last dim
  as_info.last_dim_size = out_size[as_info.dim_num];
  as_info.last_dim_stride = out_stride[as_info.dim_num];
  as_info.out_lp_step = out_size[as_info.dim_num];

  if (dims == 2) {
    as_info.dim_except_last_paras.push_back(1);
    as_info.dim_except_last_paras.push_back(out_size[0]);
    as_info.dim_except_last_paras.push_back(out_stride[0]);

    return true;
  }

  // index from left side
  for (size_t i = 0; i < dims-2; i++) {
    as_info.dim_except_last_paras.push_back(GetRangeSize(out_size, i+1, dims-2));
    as_info.dim_except_last_paras.push_back(out_size[i]);
    as_info.dim_except_last_paras.push_back(out_stride[i]);
  }
  as_info.dim_except_last_paras.push_back(1);
  as_info.dim_except_last_paras.push_back(out_size[dims-2]);
  as_info.dim_except_last_paras.push_back(out_stride[dims-2]);

  return true;
}

static bool GetOutputSizeAndStride(const string& op_type, const ge::Operator& paras,
                                   AsStridedInfo& as_info, const std::vector<int64_t>& in_shape,
                                   std::vector<int64_t>& out_size, std::vector<int64_t>& out_stride) {
  vector<int64_t> tmp_out_size;
  vector<int64_t> tmp_out_stride;
  vector<int64_t> storage_offset;

  // the parameters order in op proto is: x, size, stride, storage_offset, y
  if (!ops::GetConstIntData(paras, 1, tmp_out_size)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get const size failed!");
    return false;
  }

  if (!ops::GetConstIntData(paras, 2, tmp_out_stride)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get const stride failed!");
    return false;
  }

  if (tmp_out_size.size() != tmp_out_stride.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the dimension count of size and stride should be same!");
    return false;
  }

  if (tmp_out_size.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the dimension count should be bigger than 1!");
    return false;
  }

  if (!ops::GetConstIntData(paras, 3, storage_offset)) {
    OP_LOGD(op_type, "use storage_offset default value.");
  } else if (storage_offset[0] < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the storage_offset cannot be negative value!");
    return false;
  } else {
    as_info.storage_offset = storage_offset[0];
  }

  MergeAxis(tmp_out_size, tmp_out_stride, out_size, out_stride);
  // to make sure there are at least 2 dims in out_size and out_stride
  if (out_size.size() == 1) {
    out_size.insert(out_size.begin(), 1);
    out_stride.insert(out_stride.begin(), 0);
  }
  SetDimsTilingParas(op_type, as_info, out_size, out_stride);

  int64_t row = 0;
  int64_t col = 0;
  if (as_info.dim_num == 0) {
    row = 1;
    col = out_size[0];
  } else if (as_info.dim_num == 1) {
    row = out_size[0];
    col = out_size[1];
  } else {
    row = GetRangeSize(out_size, 0, as_info.dim_num-1);
    col = out_size[as_info.dim_num];
  }
  int64_t elem_idx_in_ori = GetElemIndexInOri(as_info, row, col);
  if (elem_idx_in_ori > GetRangeSize(in_shape, 0, in_shape.size()-1)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the output element is out of input range!");
    return false;
  }

  return true;
}

static bool GetCompileParams(const std::string& op_type, const std::vector<int64_t>& op_compile_info,
                             int64_t& max_elem_in_ub, int64_t& core_num) {
  // get compile info for vector
  OP_TILING_CHECK(
      op_compile_info.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), op_compile_info.size()),
      return false);

  max_elem_in_ub = op_compile_info[0];
  core_num = op_compile_info[1];
  OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num cannot be zero."), return false);

  OP_LOGD(op_type, "the operator info is: element_in_ub=[%ld], core_num=[%ld].", max_elem_in_ub, core_num);

  return true;
}

static void GetLpTilingParas(const std::string& op_type, const int64_t core_num,
                             const int64_t out_axis_0_lp_cnt, const int64_t out_axis_0_lp_left,
                             const int64_t out_axis_1_lp_cnt, const int64_t out_axis_1_lp_left,
                             AsStridedInfo& as_info) {
  int64_t used_core_cnt;

  if (out_axis_0_lp_cnt >= out_axis_1_lp_cnt) {
    used_core_cnt = GetCeilDiv(out_axis_0_lp_cnt, GetCeilDiv(out_axis_0_lp_cnt, core_num));
    as_info.mc_pos = 0;
    as_info.used_core_cnt = used_core_cnt;
    as_info.nlc_m_axis_0_lp_cnt = GetCeilDiv(out_axis_0_lp_cnt, used_core_cnt);
    as_info.nlc_m_axis_0_lp_left = 0;
    as_info.lc_m_axis_0_lp_cnt = out_axis_0_lp_cnt - (used_core_cnt - 1)*as_info.nlc_m_axis_0_lp_cnt;
    as_info.lc_m_axis_0_lp_left = out_axis_0_lp_left;
    as_info.core_step_in = as_info.nlc_m_axis_0_lp_cnt * as_info.m_axis_0_lp_unit;
    as_info.nlc_m_axis_1_lp_cnt = out_axis_1_lp_cnt;
    as_info.nlc_m_axis_1_lp_left = out_axis_1_lp_left;
    as_info.lc_m_axis_1_lp_cnt = out_axis_1_lp_cnt;
    as_info.lc_m_axis_1_lp_left = out_axis_1_lp_left;
  } else {
    used_core_cnt = GetCeilDiv(out_axis_1_lp_cnt, GetCeilDiv(out_axis_1_lp_cnt, core_num));
    as_info.mc_pos = 1;
    as_info.used_core_cnt = used_core_cnt;
    as_info.nlc_m_axis_1_lp_cnt = GetCeilDiv(out_axis_1_lp_cnt, used_core_cnt);
    as_info.nlc_m_axis_1_lp_left = 0;
    as_info.lc_m_axis_1_lp_cnt = out_axis_1_lp_cnt - (used_core_cnt - 1)*as_info.nlc_m_axis_1_lp_cnt;
    as_info.lc_m_axis_1_lp_left = out_axis_1_lp_left;
    as_info.core_step_in = as_info.nlc_m_axis_1_lp_cnt * as_info.m_axis_1_lp_unit;
    as_info.nlc_m_axis_0_lp_cnt = out_axis_0_lp_cnt;
    as_info.nlc_m_axis_0_lp_left = out_axis_0_lp_left;
    as_info.lc_m_axis_0_lp_cnt = out_axis_0_lp_cnt;
    as_info.lc_m_axis_0_lp_left = out_axis_0_lp_left;
  }

}

static void UpdateDimsInfo(const size_t dims, const std::vector<int64_t>& out_size, AsStridedInfo& as_info) {
  as_info.dim_num -= 1;
  as_info.dim_except_last_paras.resize(as_info.dim_num*3);
  if (dims - 2 == 1) {
    as_info.dim_except_last_paras[0] = 1;
  } else {
    for (size_t i = 0; i < dims-3; i++) {
      as_info.dim_except_last_paras[i*3] = GetRangeSize(out_size, i+1, dims-3);
    }
    as_info.dim_except_last_paras[(dims - 3)*3] = 1;
  }

}

static void SetTilingParamForLastTwoDimIsLarge(const int64_t vnc_scheme_ub_offset, const int64_t nrsecond_dim_len,
                                               const int64_t last_two_dim_len, const int64_t last_two_dim_raw_elem,
                                               const size_t dims, const std::vector<int64_t>& out_size,
                                               const std::vector<int64_t>& out_stride, AsStridedInfo& as_info) {
  as_info.tiling_mode = TILING_LAST_TWO_DIM_IS_LARGE;  // using full vnchwconv scheme, and move in 1 row per time
  as_info.out_ub_offset = vnc_scheme_ub_offset;
  as_info.m_axis_0_lp_unit = nrsecond_dim_len > VNC_ROWS ? VNC_ROWS : nrsecond_dim_len;
  as_info.m_axis_1_lp_unit = last_two_dim_len;
  as_info.m_axis_1_burst_unit = last_two_dim_raw_elem;
  as_info.rsecond_dim_size = out_size[dims-2];
  as_info.rsecond_dim_stride = out_stride[dims-2];
  as_info.out_lp_step = last_two_dim_len;

  UpdateDimsInfo(dims, out_size, as_info);

}

static void SetTilingParamForLastStrideIsZero(const int64_t vnc_scheme_ub_offset, const int64_t ele_per_block,
                                              const int64_t vnc_col_len, const int64_t nrsecond_dim_len,
                                              const int64_t max_rsecond_valid_elem, const size_t dims,
                                              const std::vector<int64_t>& out_size,
                                              const std::vector<int64_t>& out_stride, AsStridedInfo& as_info) {
  auto last_dim_len = out_size[dims - 1];
  auto rsecond_dim_len = out_size[dims - 2];
  auto rsecond_dim_stride = out_stride[dims - 2];

  if (last_dim_len < MTE_GATE * ele_per_block) {
    as_info.tiling_mode = TILING_LAST_STRIDE_IS_ZERO_SIZE_IS_SMALL;  // using full vnchwconv scheme
    as_info.m_axis_0_lp_unit = nrsecond_dim_len > VNC_ROWS ? VNC_ROWS : nrsecond_dim_len;
    auto tmp_rsecond_len = GetFloorDiv(vnc_col_len, last_dim_len);
    auto tmp_rsecond_a_row = tmp_rsecond_len > max_rsecond_valid_elem ? max_rsecond_valid_elem : tmp_rsecond_len;
    as_info.m_axis_1_lp_unit = rsecond_dim_len > tmp_rsecond_a_row ? tmp_rsecond_a_row : rsecond_dim_len;
  } else {
    as_info.tiling_mode = TILING_LAST_STRIDE_IS_ZERO_SIZE_IS_LARGE;  // using vector_dup scheme
    as_info.m_axis_0_lp_unit = 1;
    auto new_rsecond_valid_elem = GetCeilDiv(vnc_scheme_ub_offset, rsecond_dim_stride);
    as_info.m_axis_1_lp_unit = rsecond_dim_len > new_rsecond_valid_elem ? new_rsecond_valid_elem : rsecond_dim_len;
  }
  as_info.m_axis_1_burst_unit = (as_info.m_axis_1_lp_unit - 1)*rsecond_dim_stride + 1;
  as_info.out_ub_offset = vnc_scheme_ub_offset;
  as_info.rsecond_dim_size = rsecond_dim_len;
  as_info.rsecond_dim_stride = rsecond_dim_stride;
  as_info.out_lp_step = last_dim_len * rsecond_dim_len;

  UpdateDimsInfo(dims, out_size, as_info);

}

static void SetTilingParamForFirstStrideIsSmall(const int64_t vnc_scheme_ub_offset, const int64_t vnc_col_len,
                                                const int64_t ele_per_block, const int64_t nfirst_dim_len,
                                                const int64_t max_first_dim_in, const std::vector<int64_t>& out_size,
                                                const std::vector<int64_t>& out_stride, AsStridedInfo& as_info) {
  as_info.tiling_mode = TILING_FIRST_STRIDE_IS_SMALL;  // using vnchwconv scheme, and move in axis 1 in order
  as_info.out_ub_offset = vnc_scheme_ub_offset;

  auto first_dim_len = out_size[0];
  auto first_dim_stride = out_stride[0];
  auto max_valid_first_elem = GetCeilDiv(max_first_dim_in, first_dim_stride);
  as_info.m_axis_1_lp_unit = first_dim_len > max_valid_first_elem ? max_valid_first_elem : first_dim_len;
  as_info.m_axis_1_burst_unit = (as_info.m_axis_1_lp_unit - 1)*first_dim_stride + 1;

  auto burst_unit_block_align = GetCeilDiv(as_info.m_axis_1_burst_unit, ele_per_block) * ele_per_block;
  as_info.nfirst_cnt_per_row = GetDivisorAlign(GetFloorDiv(vnc_col_len, burst_unit_block_align), ele_per_block);
  auto nfirst_len_full_row = as_info.nfirst_cnt_per_row * VNC_ROWS;
  as_info.m_axis_0_lp_unit = nfirst_dim_len > nfirst_len_full_row ? nfirst_len_full_row : nfirst_dim_len;

  as_info.out_lp_step = nfirst_dim_len;
  as_info.last_dim_size = first_dim_len;
  as_info.last_dim_stride = first_dim_stride;

  // update dims info parameters
  auto dims = out_size.size();
  if (dims == 2) {
    as_info.dim_except_last_paras[1] = out_size[1];
    as_info.dim_except_last_paras[2] = out_stride[1];
  } else {
    for (size_t i = 0; i < dims-2; i++) {
    as_info.dim_except_last_paras[i*3 + 0] = GetRangeSize(out_size, i+2, dims-1);
    as_info.dim_except_last_paras[i*3 + 1] = out_size[i + 1];
    as_info.dim_except_last_paras[i*3 + 2] = out_stride[i + 1];
    }
    as_info.dim_except_last_paras[(dims - 2)*3 + 0] = 1;
    as_info.dim_except_last_paras[(dims - 2)*3 + 1] = out_size[dims-1];
    as_info.dim_except_last_paras[(dims - 2)*3 + 2] = out_stride[dims-1];

  }

}

static bool SetMultiCoreTilingParas(const std::string& op_type, const int64_t max_elem_in_ub,
                                    const int64_t core_num, const DataType& data_type,
                                    const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_size,
                                    const std::vector<int64_t>& out_stride, AsStridedInfo& as_info) {
  auto dims = out_size.size();
  if (dims < 2) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output dimension should be bigger than 2!");
    return false;
  }

  if (core_num < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "AICORE count should be bigger than 0!");
    return false;
  }

  auto last_dim_len = out_size[dims-1];
  auto nlast_dim_len = GetRangeSize(out_size, 0, dims-2);
  auto last_dim_stride = out_stride[dims-1];
  auto rsecond_dim_len = out_size[dims-2];
  auto rsecond_dim_stride = out_stride[dims-2];
  auto nfirst_dim_len = GetRangeSize(out_size, 1, dims-1);
  int64_t ele_per_block = GetFloorDiv(BYTES_PER_BLOCK, GetSizeByDataType(data_type));
  int64_t out_max_idx_in_input = GetElemIndexInOri(as_info, nlast_dim_len, last_dim_len);
  int64_t in_shape_size = GetRangeSize(in_shape, 0, in_shape.size()-1);

  int64_t vnc_scheme_ub_offset = GetDivisorAlign(GetFloorDiv(max_elem_in_ub, 2), ele_per_block);
  int64_t vnc_col_len = GetDivisorAlign(GetFloorDiv(vnc_scheme_ub_offset, VNC_ROWS), ele_per_block);
  if (ele_per_block == BYTES_PER_BLOCK) {
    vnc_col_len = GetDivisorAlign(GetFloorDiv(vnc_col_len, 2), ele_per_block);
  }
  int64_t all_in_scheme_ub_offset = GetDivisorAlign(max_elem_in_ub - vnc_col_len, ele_per_block);
  int64_t last_two_dim_len = last_dim_len * rsecond_dim_len;
  int64_t last_two_dim_raw_elem = ((last_two_dim_len - 1)/last_dim_len%rsecond_dim_len*rsecond_dim_stride +
                                   (last_two_dim_len - 1)%last_dim_len*last_dim_stride + 1);

  int64_t max_valid_elem_a_row = 1;
  if (last_dim_stride > 0) {
    max_valid_elem_a_row = GetDivisorAlign(GetCeilDiv(vnc_col_len, last_dim_stride), ele_per_block);
  }

  int64_t max_rsecond_valid_elem = 1;
  if (rsecond_dim_stride > 0) {
    int64_t tmp_value = GetCeilDiv(vnc_col_len, rsecond_dim_stride);
    max_rsecond_valid_elem = rsecond_dim_len > tmp_value ? tmp_value : rsecond_dim_len;
  }

  int64_t max_first_dim_in = GetDivisorAlign(GetFloorDiv(vnc_col_len, ele_per_block), ele_per_block);

  as_info.vnc_col_size = vnc_col_len;
  if (last_dim_len >= MTE_GATE * ele_per_block && last_dim_stride == 1) {
    as_info.tiling_mode = TILING_LAST_STRIDE_IS_ONE;  // using data_move scheme, and move in 1 row per time
    as_info.out_ub_offset = 0;
    as_info.m_axis_1_lp_unit = last_dim_len > vnc_col_len ? vnc_col_len : last_dim_len;
    int64_t axis_1_unit_block_align = GetCeilDiv(as_info.m_axis_1_lp_unit, ele_per_block) * ele_per_block;
    int64_t tmp_m_axis_0_lp_unit = GetFloorDiv(max_elem_in_ub, axis_1_unit_block_align);
    as_info.m_axis_0_lp_unit = nlast_dim_len > tmp_m_axis_0_lp_unit ? tmp_m_axis_0_lp_unit : nlast_dim_len;
    as_info.m_axis_1_burst_unit = (as_info.m_axis_1_lp_unit - 1) * last_dim_stride + 1;

  } else if (max_valid_elem_a_row >= MTE_GATE * ele_per_block && last_dim_len >= MTE_GATE * ele_per_block) {
    as_info.tiling_mode = TILING_LAST_DIM_IS_LARGE;  // using full vnchwconv scheme, and move in 1 row per time
    as_info.out_ub_offset = vnc_scheme_ub_offset;
    as_info.m_axis_0_lp_unit = nlast_dim_len > VNC_ROWS ? VNC_ROWS : nlast_dim_len;
    as_info.m_axis_1_lp_unit = last_dim_len > max_valid_elem_a_row ? max_valid_elem_a_row : last_dim_len;
    as_info.m_axis_1_burst_unit = (as_info.m_axis_1_lp_unit - 1) * last_dim_stride + 1;

  } else if (dims > 2 && last_two_dim_raw_elem <= vnc_col_len &&
             last_two_dim_len >= MTE_GATE * ele_per_block && last_two_dim_len <= vnc_col_len) {
    auto nrsecond_dim_len = GetRangeSize(out_size, 0, dims-3);
    SetTilingParamForLastTwoDimIsLarge(vnc_scheme_ub_offset, nrsecond_dim_len, last_two_dim_len,
                                       last_two_dim_raw_elem, dims, out_size, out_stride, as_info);
    // for multiple core parameters
    nlast_dim_len = nrsecond_dim_len;
    last_dim_len = last_two_dim_len;

  } else if (max_valid_elem_a_row >= MTE_GATE * ele_per_block && last_dim_len < MTE_GATE * ele_per_block) {
    as_info.tiling_mode = TILING_LAST_DIM_IS_SMALL;  // using single vnchwconv scheme, and move in 1 row per time
    as_info.out_ub_offset = vnc_scheme_ub_offset;
    int64_t tmp_elems_in_count = (last_dim_len - 1) * last_dim_stride + 1;
    int64_t tmp_elmes_in_block_align = GetCeilDiv(tmp_elems_in_count, ele_per_block) * ele_per_block;
    int64_t tmp_axis_0_lp_unit = GetFloorDiv(vnc_col_len, tmp_elmes_in_block_align);
    as_info.m_axis_0_lp_unit = nlast_dim_len > tmp_axis_0_lp_unit ? tmp_axis_0_lp_unit : nlast_dim_len;
    as_info.m_axis_1_lp_unit = last_dim_len;
    as_info.m_axis_1_burst_unit = (as_info.m_axis_1_lp_unit - 1) * last_dim_stride + 1;

  } else if (out_max_idx_in_input <= all_in_scheme_ub_offset || in_shape_size <= all_in_scheme_ub_offset) {
    as_info.tiling_mode = TILING_INPUT_OR_OUTPUT_IS_ALL_IN;  // using scalar scheme, and move all in
    as_info.out_ub_offset = all_in_scheme_ub_offset;
    as_info.m_axis_1_lp_unit = last_dim_len > vnc_col_len ? vnc_col_len : last_dim_len;
    int64_t tmp_m_axis_0_lp_unit = GetFloorDiv(vnc_col_len, as_info.m_axis_1_lp_unit);
    as_info.m_axis_0_lp_unit = nlast_dim_len > tmp_m_axis_0_lp_unit ? tmp_m_axis_0_lp_unit : nlast_dim_len;

    if (out_max_idx_in_input <= all_in_scheme_ub_offset) {
      as_info.m_axis_1_burst_unit = out_max_idx_in_input;
    } else {
      as_info.m_axis_1_burst_unit = in_shape_size;
    }

  } else if (dims > 2 && last_dim_stride == 0 &&
             max_rsecond_valid_elem * last_dim_len >= MTE_GATE * ele_per_block && max_rsecond_valid_elem >= 2) {
    auto nrsecond_dim_len = GetRangeSize(out_size, 0, dims-3);
    SetTilingParamForLastStrideIsZero(vnc_scheme_ub_offset, ele_per_block, vnc_col_len, nrsecond_dim_len,
                                      max_rsecond_valid_elem, dims, out_size, out_stride, as_info);
    // for multiple core parameters
    nlast_dim_len = nrsecond_dim_len;
    last_dim_len = out_size[dims-2];

  } else if (max_first_dim_in > out_stride[0] && out_stride[0] > 0 && nfirst_dim_len >= MTE_GATE * ele_per_block) {
    // process the first dimension from left side as the last
    SetTilingParamForFirstStrideIsSmall(vnc_scheme_ub_offset, vnc_col_len, ele_per_block, nfirst_dim_len,
                                        max_first_dim_in, out_size, out_stride, as_info);
    nlast_dim_len = nfirst_dim_len;
    last_dim_len = out_size[0];
  } else {
    as_info.out_ub_offset = vnc_scheme_ub_offset;
    int64_t vnc_col_valid_elem = GetDivisorAlign(GetFloorDiv(vnc_col_len, ele_per_block), ele_per_block);
    as_info.m_axis_1_lp_unit = last_dim_len > vnc_col_valid_elem ? vnc_col_valid_elem : last_dim_len;

    if (last_dim_len >= vnc_col_valid_elem) {
      // using full vnchwconv scheme, and move in 1 block per time
      as_info.tiling_mode = TILING_LAST_LARGE_DIM_LARGE_STRIDE;
      as_info.m_axis_0_lp_unit = nlast_dim_len > VNC_ROWS ? VNC_ROWS : nlast_dim_len;
    } else {
      // using single vnchwconv scheme, and move in 1 block per time
      as_info.tiling_mode = TILING_LAST_SMALL_DIM_LARGE_STRIDE;
      as_info.m_axis_0_lp_unit = GetFloorDiv(vnc_col_valid_elem, as_info.m_axis_1_lp_unit);
    }
    as_info.m_axis_1_burst_unit = 1;

  }

  int64_t out_axis_0_lp_cnt = GetCeilDiv(nlast_dim_len, as_info.m_axis_0_lp_unit);
  int64_t out_axis_0_lp_left = nlast_dim_len % as_info.m_axis_0_lp_unit;
  int64_t out_axis_1_lp_cnt = GetCeilDiv(last_dim_len, as_info.m_axis_1_lp_unit);
  int64_t out_axis_1_lp_left = last_dim_len % as_info.m_axis_1_lp_unit;
  GetLpTilingParas(op_type, core_num, out_axis_0_lp_cnt, out_axis_0_lp_left,
                   out_axis_1_lp_cnt, out_axis_1_lp_left, as_info);

  return true;
}

static void Serialize(utils::OpRunInfo& run_info, const AsStridedInfo& as_info) {
  run_info.AddTilingData(as_info.tiling_mode);
  run_info.AddTilingData(as_info.used_core_cnt);
  run_info.AddTilingData(as_info.out_ub_offset);
  run_info.AddTilingData(as_info.vnc_col_size);
  run_info.AddTilingData(as_info.m_axis_1_burst_unit);
  run_info.AddTilingData(as_info.m_axis_1_lp_unit);
  run_info.AddTilingData(as_info.m_axis_0_lp_unit);
  run_info.AddTilingData(as_info.mc_pos);
  run_info.AddTilingData(as_info.core_step_in);
  run_info.AddTilingData(as_info.nlc_m_axis_1_lp_cnt);
  run_info.AddTilingData(as_info.nlc_m_axis_1_lp_left);
  run_info.AddTilingData(as_info.lc_m_axis_1_lp_cnt);
  run_info.AddTilingData(as_info.lc_m_axis_1_lp_left);
  run_info.AddTilingData(as_info.nlc_m_axis_0_lp_cnt);
  run_info.AddTilingData(as_info.nlc_m_axis_0_lp_left);
  run_info.AddTilingData(as_info.lc_m_axis_0_lp_cnt);
  run_info.AddTilingData(as_info.lc_m_axis_0_lp_left);
  run_info.AddTilingData(as_info.storage_offset);
  run_info.AddTilingData(as_info.last_dim_size);
  run_info.AddTilingData(as_info.last_dim_stride);
  run_info.AddTilingData(as_info.rsecond_dim_size);
  run_info.AddTilingData(as_info.rsecond_dim_stride);
  run_info.AddTilingData(as_info.out_lp_step);
  run_info.AddTilingData(as_info.nfirst_cnt_per_row);
  run_info.AddTilingData(as_info.dim_num);
  for (int64_t i = 0; i < static_cast<int64_t>(as_info.dim_except_last_paras.size()); i++) {
    run_info.AddTilingData(as_info.dim_except_last_paras[i]);
  }
  run_info.SetBlockDim(as_info.used_core_cnt);
}

static std::string GetVectorData(const AsStridedInfo& as_info) {
  std::string dims_info = "the dims info is:";
  for (int64_t i = 0; i < static_cast<int64_t>(as_info.dim_except_last_paras.size()); i++) {
    dims_info += " " + std::to_string(as_info.dim_except_last_paras[i]);
  }

  return dims_info;
}

static void PrintTilingParas(const std::string& op_type, const AsStridedInfo& as_info) {
  OP_LOGD(op_type, "tiling_mode=%ld", as_info.tiling_mode);
  OP_LOGD(op_type, "used_core_cnt=%ld", as_info.used_core_cnt);
  OP_LOGD(op_type, "out_ub_offset=%ld", as_info.out_ub_offset);
  OP_LOGD(op_type, "vnc_col_size=%ld", as_info.vnc_col_size);
  OP_LOGD(op_type, "m_axis_1_burst_unit=%ld", as_info.m_axis_1_burst_unit);
  OP_LOGD(op_type, "m_axis_1_lp_unit=%ld", as_info.m_axis_1_lp_unit);
  OP_LOGD(op_type, "m_axis_0_lp_unit=%ld", as_info.m_axis_0_lp_unit);
  OP_LOGD(op_type, "mc_pos=%ld", as_info.mc_pos);
  OP_LOGD(op_type, "core_step_in=%ld", as_info.core_step_in);
  OP_LOGD(op_type, "nlc_m_axis_1_lp_cnt=%ld", as_info.nlc_m_axis_1_lp_cnt);
  OP_LOGD(op_type, "nlc_m_axis_1_lp_left=%ld", as_info.nlc_m_axis_1_lp_left);
  OP_LOGD(op_type, "lc_m_axis_1_lp_cnt=%ld", as_info.lc_m_axis_1_lp_cnt);
  OP_LOGD(op_type, "lc_m_axis_1_lp_left=%ld", as_info.lc_m_axis_1_lp_left);
  OP_LOGD(op_type, "nlc_m_axis_0_lp_cnt=%ld", as_info.nlc_m_axis_0_lp_cnt);
  OP_LOGD(op_type, "nlc_m_axis_0_lp_left=%ld", as_info.nlc_m_axis_0_lp_left);
  OP_LOGD(op_type, "lc_m_axis_0_lp_cnt=%ld", as_info.lc_m_axis_0_lp_cnt);
  OP_LOGD(op_type, "lc_m_axis_0_lp_left=%ld", as_info.lc_m_axis_0_lp_left);
  OP_LOGD(op_type, "storage_offset=%ld", as_info.storage_offset);
  OP_LOGD(op_type, "last_dim_size=%ld", as_info.last_dim_size);
  OP_LOGD(op_type, "last_dim_stride=%ld", as_info.last_dim_stride);
  OP_LOGD(op_type, "rsecond_dim_size=%ld", as_info.rsecond_dim_size);
  OP_LOGD(op_type, "rsecond_dim_stride=%ld", as_info.rsecond_dim_stride);
  OP_LOGD(op_type, "out_lp_step=%ld", as_info.out_lp_step);
  OP_LOGD(op_type, "nfirst_cnt_per_row=%ld", as_info.nfirst_cnt_per_row);
  OP_LOGD(op_type, "dim_num=%ld", as_info.dim_num);
  OP_LOGD(op_type, "%s", GetVectorData(as_info).c_str());
}

bool AsStridedTiling(const std::string& op_type,
                     const ge::Operator& op_paras,
                     const std::vector<int64_t>& op_compile_info,
                     utils::OpRunInfo& run_info) {
  OP_LOGI(op_type, "Tiling is running.");
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op info failed.");
    return false;
  }

  auto input_desc = operator_info->MutableInputDesc(0);
  if (input_desc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input desc failed.");
    return false;
  }

  std::vector<int64_t> in_shape = input_desc->MutableShape().GetDims();
  auto data_type = input_desc->GetDataType();

  int64_t max_elem_in_ub = 1;
  int64_t core_num = 1;
  if (!GetCompileParams(op_type, op_compile_info, max_elem_in_ub, core_num)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile information failed.");
    return false;
  }

  AsStridedInfo as_info;
  vector<int64_t> out_size;
  vector<int64_t> out_stride;
  if (!GetOutputSizeAndStride(op_type, op_paras, as_info, in_shape, out_size, out_stride)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get output size and stride failed.");
    return false;
  }

  // set multiple core tiling parameters
  if (!SetMultiCoreTilingParas(op_type, max_elem_in_ub, core_num, data_type,
                               in_shape, out_size, out_stride, as_info)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get tiling parameters failed.");
    return false;
  }
  // send tiling parameters to operator
  Serialize(run_info, as_info);
  // print tiling parameters for debug
  PrintTilingParas(op_type, as_info);

  return true;
}

REGISTER_OP_TILING_V3_WITH_VECTOR(AsStrided, AsStridedTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);

};
