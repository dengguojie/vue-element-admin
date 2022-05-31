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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_OP_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_OP_IMPL_H_
#include <cstdint>
#include "register/op_compile_info_base.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"

namespace optiling {
struct TransDataMode1010Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t used_core_cnt;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t dst_cl_lp_step_in;
  int64_t dst_cl_lp_step_out;
  int64_t dst_cl_step_in;
  int64_t dst_cl_step_out;
  int64_t dst_cr_lp_step_in;
  int64_t dst_cr_lp_step_out;
  int64_t dst_cr_step_in;
  int64_t nc_le_vcol;
  int64_t vnc_line_size;
  int64_t pln_dst_cl_size;
  int64_t pln_dst_cr_size;
  int64_t vnc_row_size;
  int64_t c_lp_step_in;
  int64_t c_lp_step_out;
  int64_t c_step_out;
  int64_t c0_size;
  int64_t c_mod_c0;
  int64_t c_lp_unit;
  int64_t nlc_dst_cl_lp_cnt;
  int64_t nlc_vnc_row_cl_left;
  int64_t nlc_last_line_cl_cnt;
  int64_t nlc_dst_cr_lp_cnt;
  int64_t nlc_vnc_row_left;
  int64_t nlc_last_line_cr_cnt;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_c_left;
  int64_t lc_dst_cl_lp_cnt;
  int64_t lc_vnc_row_cl_left;
  int64_t lc_last_line_cl_cnt;
  int64_t lc_dst_cr_lp_cnt;
  int64_t lc_vnc_row_left;
  int64_t lc_last_line_cr_cnt;
  int64_t lc_c_lp_cnt;
  int64_t lc_c_left;
};

struct TransDataTc201Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t mc_pos;
  int64_t used_core_cnt;
  int64_t src_r2nd_dst_r2nd_same;
  int64_t c0_len;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t nlc_dst_r2nd_lp_cnt;
  int64_t nlc_src_cl_lp_cnt;
  int64_t nlc_src_left_lp_cnt;
  int64_t nlc_dst_r2nd_left;
  int64_t nlc_src_cl_left;
  int64_t nlc_src_left_left;
  int64_t lc_dst_r2nd_lp_cnt;
  int64_t lc_src_cl_lp_cnt;
  int64_t lc_src_left_lp_cnt;
  int64_t lc_dst_r2nd_left;
  int64_t lc_src_cl_left;
  int64_t lc_src_left_left;
  int64_t dst_r2nd_lp_unit;
  int64_t dst_r2nd_step_in;
  int64_t dst_r2nd_step_out;
  int64_t dst_r2nd_lp_step_in;
  int64_t dst_r2nd_lp_step_out;
  int64_t src_cl_lp_unit;
  int64_t all_c_in;
  int64_t src_cl_step_in;
  int64_t src_cl_step_out;
  int64_t src_cl_lp_step_in;
  int64_t src_cl_lp_step_out;
  int64_t c_mod_c0;
  int64_t src_left_lp_unit;
  int64_t src_left_step_in;
  int64_t src_left_step_out;
  int64_t src_left_lp_step_in;
  int64_t src_left_lp_step_out;
  int64_t dst_r2nd_in_0_size;
  int64_t dst_r2nd_in_0_src_rsize;
  int64_t dst_r2nd_in_0_src_asize;
  int64_t dst_r2nd_in_1_size;
  int64_t dst_r2nd_in_1_src_rsize;
  int64_t dst_r2nd_in_1_src_asize;
  int64_t dst_r2nd_dims;
  int64_t vnc_col_size;
  int64_t all_r2nd_in;
};

struct TransDataNtc100Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t mc_pos;
  int64_t used_core_cnt;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t vnc_line_size;
  int64_t c_mod_c0;
  int64_t c0_size;
  int64_t cl_dims;
  int64_t cr_dims;
  int64_t r1st_src_r2nd_dst_same;
  int64_t src_cl_step_in;
  int64_t src_cl_step_out;
  int64_t src_cl_lp_unit;
  int64_t src_cl_lp_step_in;
  int64_t src_cl_lp_step_out;
  int64_t src_c_step_in;
  int64_t src_c_lp_unit;
  int64_t src_c_lp_step_in;
  int64_t src_c_lp_step_out;
  int64_t src_cr_step_in;
  int64_t src_cr_step_out;
  int64_t src_cr_lp_unit;
  int64_t src_cr_lp_step_in;
  int64_t src_cr_lp_step_out;

  int64_t nlc_cl_lp_cnt;
  int64_t nlc_cl_left;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_c_left;
  int64_t nlc_cr_lp_cnt;
  int64_t nlc_cr_left;
  int64_t lc_cl_lp_cnt;
  int64_t lc_cl_left;
  int64_t lc_c_lp_cnt;
  int64_t lc_c_left;
  int64_t lc_cr_lp_cnt;
  int64_t lc_cr_left;
  int64_t cl_out_idx_0_size;
  int64_t cl_out_idx_0_dst_rsize;
  int64_t cl_out_idx_0_dst_asize;
  int64_t cl_out_idx_1_size;
  int64_t cl_out_idx_1_dst_rsize;
  int64_t cl_out_idx_1_dst_asize;
  int64_t cr_out_idx_0_size;
  int64_t cr_out_idx_0_dst_rsize;
  int64_t cr_out_idx_0_dst_asize;
  int64_t cr_out_idx_1_size;
  int64_t cr_out_idx_1_dst_rsize;
  int64_t cr_out_idx_1_dst_asize;
};

struct TransDataMode1011Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t used_core_cnt;
  int64_t mc_on_cl;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t dst_r2nd_lp_step_in;
  int64_t dst_r2nd_lp_step_out;
  int64_t dst_r2nd_step_in;
  int64_t dst_r2nd_lp_unit;
  int64_t src_cl_lp_step_in;
  int64_t vnc_line_size;
  int64_t src_cl_lp_unit;
  int64_t src_cl_lp_step_out;
  int64_t c_lp_step_in;
  int64_t c_lp_step_out;
  int64_t c_step_out;
  int64_t c0_size;
  int64_t c_mod_c0;
  int64_t c_lp_unit;
  int64_t nlc_dst_r2nd_lp_cnt;
  int64_t nlc_dst_r2nd_left;
  int64_t nlc_src_cl_lp_cnt;
  int64_t nlc_src_cl_left;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_c_left;
  int64_t lc_dst_r2nd_lp_cnt;
  int64_t lc_dst_r2nd_left;
  int64_t lc_src_cl_lp_cnt;
  int64_t lc_src_cl_left;
  int64_t lc_c_lp_cnt;
  int64_t lc_c_left;
  int64_t cl_out_0_size;
  int64_t cl_out_0_src_rsize;
  int64_t cl_out_0_dst_asize;
  int64_t cl_out_1_size;
  int64_t cl_out_1_src_rsize;
  int64_t cl_out_1_dst_asize;
};

struct TransDataNtc200Param {
  int64_t tiling_mode;
  int64_t ub_offset;
  int64_t mc_pos;
  int64_t used_core_cnt;
  int64_t c0_len;
  int64_t core_step_in;
  int64_t core_step_out;
  int64_t nlc_cr_lp_cnt;
  int64_t nlc_c_lp_cnt;
  int64_t nlc_cl_lp_cnt;
  int64_t nlc_cr_left;
  int64_t nlc_c_left;
  int64_t nlc_cl_left;
  int64_t lc_cr_lp_cnt;
  int64_t lc_c_lp_cnt;
  int64_t lc_cl_lp_cnt;
  int64_t lc_cr_left;
  int64_t lc_c_left;
  int64_t lc_cl_left;
  int64_t dst_cr_lp_unit;
  int64_t src_c_lp_unit;
  int64_t dst_cl_lp_unit;
  int64_t vnc_col_size;
  int64_t dst_cr_step_in;
  int64_t dst_cr_step_out;
  int64_t dst_cr_lp_step_in;
  int64_t dst_cr_lp_step_out;
  int64_t dst_c_size;
  int64_t src_c_step_in;
  int64_t src_c_step_out;
  int64_t src_c_lp_step_in;
  int64_t src_c_lp_step_out;
  int64_t dst_cr_all_in;
  int64_t dst_cl_step_in;
  int64_t dst_cl_step_out;
  int64_t dst_cl_lp_step_in;
  int64_t dst_cl_lp_step_out;
  int64_t c_mod_c0;
  int64_t dst_cr_dims;
  int64_t dst_cl_dims;
  int64_t is_mc_cr;
  int64_t is_mc_cl;
  int64_t src_r2nd_dst_r1st_same;
  int64_t left_cl_c_cr_size;

  int64_t cl_in_idx_0_size;
  int64_t cl_in_idx_0_dst_rsize;
  int64_t cl_in_idx_0_src_asize;
  int64_t cl_in_idx_1_size;
  int64_t cl_in_idx_1_dst_rsize;
  int64_t cl_in_idx_1_src_asize;
  int64_t cr_in_idx_0_size;
  int64_t cr_in_idx_0_dst_rsize;
  int64_t cr_in_idx_0_src_asize;
  int64_t cr_in_idx_1_size;
  int64_t cr_in_idx_1_dst_rsize;
  int64_t cr_in_idx_1_src_asize;
};

template <size_t SRC, size_t DST, typename T>
class TableDriven2 {
 public:
  explicit TableDriven2(const T& default_value) : default_value_(default_value) {
    for (size_t i = 0; i < SRC; ++i) {
      for (size_t j = 0; j < DST; ++j) {
        elements[i][j] = default_value;
      }
    }
  }
  T Find(size_t src, size_t dst) const {
    if (src >= SRC || dst >= DST) {
      return default_value_;
    }
    return elements[src][dst];
  }
  const T* FindPointer(size_t src, size_t dst) const {
    if (src >= SRC || dst >= DST) {
      return nullptr;
    }
    return &elements[src][dst];
  }
  template <typename... Arg>
  TableDriven2& Add(size_t src, size_t dst, const Arg&... arg) {
    auto& element = elements[src][dst];
    element = T(arg...);
    return *this;
  }

  template <typename... Arg>
  TableDriven2& Add(size_t* src, size_t src_len, size_t dst, const Arg&... arg) {
    for (size_t i = 0; i < src_len; ++i) {
      auto& element = elements[src[i]][dst];
      element = T(arg...);
    }
    return *this;
  }

  template <typename... Arg>
  TableDriven2& Add(size_t src, size_t* dst, size_t dst_len, const Arg&... arg) {
    for (size_t i = 0; i < dst_len; ++i) {
      auto& element = elements[src][dst[i]];
      element = T(arg...);
    }
    return *this;
  }

 private:
  T default_value_;
  T elements[SRC][DST];
};
struct TransDataCompileInfo : public optiling::CompileInfoBase {
  int64_t ub_size;
  int64_t block_dim;
  int64_t group;
  int64_t vnc_fp32_flag;
};

namespace transdata {
constexpr int32_t DIM_IDX_NEG_ONE = -1;
constexpr int32_t DIM_IDX_NEG_TWO = -2;
constexpr size_t DIM_NUM_2 = 2;
constexpr size_t DIM_IDX_2 = 2;
constexpr size_t DIM_IDX_3 = 3;
constexpr size_t DIM_IDX_4 = 4;
constexpr size_t DIM_IDX_5 = 5;
constexpr size_t ARRAYNDLEN = 3;
constexpr size_t ARRAY4DLEN = 2;
constexpr int64_t BLOCK_BYTE_SIZE = 32;
constexpr int64_t NI_16 = 16;
constexpr int64_t C0_16 = 16;
constexpr int64_t C0_32 = 32;
constexpr int64_t VNC_LINES = 16;

constexpr size_t FORMAT_LEN_2D = 2;
constexpr size_t SHAPE_LEN_2D = 2;
constexpr size_t SHAPE_LEN_4D = 4;
constexpr size_t SHAPE_LEN_5D = 5;
constexpr size_t SHAPE_LEN_6D = 6;
constexpr size_t SHAPE_LEN_CAPACITY_SIZE = 8;

constexpr int64_t TILING_MODE_2001 = 2001;
constexpr int64_t TILING_MODE_2002 = 2002;
constexpr int64_t TILING_MODE_2003 = 2003;
constexpr int64_t TILING_MODE_2010 = 2010;
constexpr int64_t TILING_MODE_2011 = 2011;
constexpr int64_t TILING_MODE_2012 = 2012;
constexpr int64_t TILING_MODE_1000 = 1000;
constexpr int64_t TILING_MODE_1001 = 1001;
constexpr int64_t TILING_MODE_1010 = 1010;
constexpr int64_t TILING_MODE_1011 = 1011;

constexpr int64_t TRANSDATA_TILING_FACTOR_2 = 2;
constexpr int64_t TRANSDATA_TILING_FACTOR_4 = 4;
constexpr int64_t TRANSDATA_TILING_FACTOR_8 = 8;
constexpr int64_t TRANSDATA_TILING_FACTOR_16 = 16;
constexpr int64_t TRANSDATA_TILING_FACTOR_15 = 15;
constexpr int64_t TRANSDATA_TILING_FACTOR_54 = 54;
constexpr int64_t TRANSDATA_TILING_FACTOR_56 = 56;

constexpr int64_t TRANSDATA_TILING_PARAM_2 = 2;
constexpr int64_t TRANSDATA_TILING_PARAM_3 = 3;
constexpr int64_t TRANSDATA_TILING_PARAM_4 = 4;
constexpr int64_t TRANSDATA_TILING_PARAM_5 = 5;
constexpr int64_t TRANSDATA_TILING_PARAM_31 = 31;
constexpr int64_t TRANSDATA_TILING_PARAM_63 = 63;
constexpr int64_t TRANSDATA_TILING_PARAM_127 = 127;

enum RealAxisType {
  RAT_C,
  RAT_H,
  RAT_N,
  RAT_T,
  RAT_D,

  RAT_END
};
enum RealFormat {
  RF_NHC,
  RF_NCH,
  RF_NCHT,
  RF_HNC,
  RF_HCNT,
  RF_NCDH,
  RF_NDCHT,
  RF_NDHC,
  RF_HCN,
  RF_CHNT,
  RF_DHCN,
  RF_DCHNT,

  RF_END
};

const std::map<RealFormat, std::string> RealFormatToStringMap = {
    {RF_NHC, "NHC"},   {RF_NCH, "NCH"},   {RF_NCHT, "NCHT"},   {RF_HNC, "HNC"},
    {RF_HCNT, "HCNT"}, {RF_NCDH, "NCDH"}, {RF_NDCHT, "NDCHT"}, {RF_NDHC, "NDHC"},
    {RF_HCN, "HCN"},   {RF_CHNT, "CHNT"}, {RF_DHCN, "DHCN"},   {RF_DCHNT, "DCHNT"}};
struct RealFormatAxisType {
  int32_t rank;
  RealAxisType axis_type[8];
};

inline RealAxisType GetAxisType(RealFormat real_format, int32_t dim_index) {
  static RealFormatAxisType real_formats_axis_types[RF_END] = {
      {3, {RAT_N, RAT_H, RAT_C}},                // NHC
      {3, {RAT_N, RAT_C, RAT_H}},                // NCH
      {4, {RAT_N, RAT_C, RAT_H, RAT_T}},         // NCHT
      {3, {RAT_H, RAT_N, RAT_C}},                // HNC
      {4, {RAT_H, RAT_C, RAT_N, RAT_T}},         // HCNT
      {4, {RAT_N, RAT_C, RAT_D, RAT_H}},         // NCDH
      {5, {RAT_N, RAT_D, RAT_C, RAT_H, RAT_T}},  // NDCHT
      {4, {RAT_N, RAT_D, RAT_H, RAT_C}},         // NDHC
      {3, {RAT_H, RAT_C, RAT_N}},                // HCN
      {4, {RAT_C, RAT_H, RAT_N, RAT_T}},         // CHNT
      {4, {RAT_D, RAT_H, RAT_C, RAT_N}},         // DHCN
      {5, {RAT_D, RAT_C, RAT_H, RAT_N, RAT_T}},  // DCHNT
  };
  if (real_format >= RF_END) {
    return RAT_END;
  }
  auto& axis_types = real_formats_axis_types[real_format];

  if (dim_index < 0) {
    if (axis_types.rank + dim_index < 0) {
      return RAT_END;
    }
    return axis_types.axis_type[axis_types.rank + dim_index];
  } else {
    if (dim_index >= axis_types.rank) {
      return RAT_END;
    }
    return axis_types.axis_type[dim_index];
  }
}
inline int32_t GetAxisIndex(RealFormat real_format, RealAxisType axis_type) {
  static int32_t real_formats_axis_index[RF_END][RAT_END] = {
      // C, H, N, T, D
      {2, 1, 0, -1, -1},  // NHC
      {1, 2, 0, -1, -1},  // NCH
      {1, 2, 0, 3, -1},   // NCHT
      {2, 0, 1, -1, -1},  // HNC
      {1, 0, 2, 3, -1},   // HCNT
      {1, 3, 0, -1, 2},   // NCDH
      {2, 3, 0, 4, 1},    // NDCHT
      {3, 2, 0, -1, 1},   // NDHC
      {1, 0, 2, -1, -1},  // HCN
      {0, 1, 2, 3, -1},   // CHNT
      {2, 1, 3, -1, 0},   // DHCN
      {1, 2, 3, 4, 0},    // DCHNT
  };
  if (real_format >= RF_END) {
    return -1;
  }
  if (axis_type >= RAT_END) {
    return -1;
  }
  return real_formats_axis_index[real_format][axis_type];
}

struct RealSrcDstFormat {
  RealSrcDstFormat() = default;
  RealSrcDstFormat(const RealSrcDstFormat& other) = default;
  RealSrcDstFormat& operator=(const RealSrcDstFormat& other) = default;
  RealSrcDstFormat(RealFormat src, RealFormat dst) : src(src), dst(dst) {
  }
  RealFormat src;
  RealFormat dst;
};

using RealShapeConvertFunc = bool (*)(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                                      gert::Shape& real_in_shape, gert::Shape& real_out_shape);
RealShapeConvertFunc GetRealShapeConvertFunc(ge::Format src_format, ge::Format dst_format);

using DoRealTilingFunc = ge::graphStatus (*)(gert::TilingContext* context, const gert::Shape& in_shape,
                                             const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                             const TransDataCompileInfo* compile_info);
DoRealTilingFunc GetRealTilingFunc(RealFormat src_rf, RealFormat dst_rf);

ge::graphStatus TillingPositiveMode1010(gert::TilingContext* context, const gert::Shape& in_shape,
                                        const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                        const TransDataCompileInfo* compile_info);

ge::graphStatus TilingNegativeTc201(gert::TilingContext* context, const gert::Shape& in_shape,
                                    const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                    const TransDataCompileInfo* compile_info);

ge::graphStatus TilingPositiveSourceNtc100(gert::TilingContext* context, const gert::Shape& in_shape,
                                           const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                           const TransDataCompileInfo* compile_info);

ge::graphStatus TillingPositiveMode1011(gert::TilingContext* context, const gert::Shape& in_shape,
                                        const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                        const TransDataCompileInfo* compile_info);

ge::graphStatus TilingNegativeNtc200(gert::TilingContext* context, const gert::Shape& in_shape,
                                     const gert::Shape& out_shape, const RealSrcDstFormat* real_formats,
                                     const TransDataCompileInfo* compile_info);

int64_t GetShapeSize(const gert::Shape& in_shape, int32_t pos);

int64_t GetC0SizeWithType(ge::DataType& dtype);
}  // namespace transdata
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_OP_IMPL_H_
