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
 * \file trans_data.cc
 * \brief dynamic TransData op tiling
 */
#include <string>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "trans_data_common.h"
#include "transpose.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
using namespace ge;

int64_t GetC0SizeWithType(DataType& dtype) {
  if (dtype == DT_INT8 || dtype == DT_UINT8) {
    return C0_32;
  }
  return C0_16;
}

bool CheckTensorShape(const std::string& op_type, int64_t ub_size, int64_t block_dim, std::vector<int64_t> out_shape) {
  int32_t out_dims = out_shape.size();

  if (ub_size < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [TransDataTiling] : CheckTensorShape, ub_size is invalid.");
    return false;
  }

  if (block_dim < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [TransDataTiling] : CheckTensorShape, block_dim is invalid.");
    return false;
  }

  if (out_dims == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [TransDataTiling] : CheckTensorShape, out_shape is invalid.");
    return false;
  }

  for (int32_t i = 0; i < out_dims; i++) {
    if (out_shape[i] <= 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                      "op [TransDataTiling] : CheckTensorShape, out_shape.shape[i] must be > 0");
      return false;
    }
  }

  return true;
}

bool GetCompileParams(const nlohmann::json& compile_info_json, int64_t& ub_size, int64_t& block_dim,
                      int64_t& group, int64_t& vnc_fp32_flag, const std::string& op_type) {
  using namespace nlohmann;

  auto all_vars = compile_info_json["vars"];

  if (all_vars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get ub_size error");
    return false;
  }
  ub_size = all_vars["ub_size"].get<std::int64_t>();

  if (all_vars.count("block_dim") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get block_dim error");
    return false;
  }
  block_dim = all_vars["block_dim"].get<std::int64_t>();

  if (all_vars.count("group") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get group error");
    return false;
  }
  group = all_vars["group"].get<std::int64_t>();

  if (block_dim == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "Core count cannot be zero!");
    return false;
  }

  if (all_vars.count("vnc_fp32_flag") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, cannot get vnc_fp32_flag");
  } else {
    vnc_fp32_flag = all_vars["vnc_fp32_flag"].get<std::int64_t>();
  }

  OP_LOGD(op_type, "GetCompileParams, \
          ub_size[%d], block_dim[%d], group[%d], vnc_fp32_flag[%d].",
          ub_size, block_dim, group, vnc_fp32_flag);

  return true;
}

bool GetRenew2Shape(std::vector<int64_t> in_shape, std::vector<int64_t> out_shape, ge::Format& src_format,
                    ge::Format& dst_format, int64_t c0_len, int64_t group, std::vector<int64_t>& in_shape_new,
                    std::vector<int64_t>& out_shape_new, std::string& real_src_format, std::string& real_dst_format) {

  if ((src_format == FORMAT_NCHW || src_format == FORMAT_NHWC) && (dst_format == FORMAT_NC1HWC0)) {
    int64_t hw_idx = GetIdxFromFormat(HW_IDX_MAP, src_format);
    int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
    real_dst_format = "NCHT";
    if (src_format == FORMAT_NCHW) {
      real_src_format = "NCH";
      for (size_t i = 0; i < in_shape.size() - hw_idx; i++) {
        in_shape_new.push_back(in_shape[i]);
      }
      int64_t last_size = GetShapeSize(in_shape, hw_idx);
      in_shape_new.push_back(last_size);
    } else {
      if (in_shape.size() < 1) {
        VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, in_shape size < 1");
        return false;
      }
      real_src_format = "NHC";
      for (int64_t i = 0; i < hw_idx; i++) {
        in_shape_new.push_back(in_shape[i]);
      }
      int64_t n = in_shape.size() - 1;
      int64_t shape_size = 1;
      for (int64_t i = hw_idx; i < n; i++) {
        shape_size *= in_shape[i];
      }
      in_shape_new.push_back(shape_size);
      in_shape_new.push_back(in_shape[in_shape.size() - 1]);
    }
    int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
    int64_t axis_n = in_shape[0];
    int64_t axis_h = in_shape_new[hw_idx];
    int64_t axis_c0 = c0_len;
    out_shape_new.push_back(axis_n);
    out_shape_new.push_back(axis_c1);
    out_shape_new.push_back(axis_h);
    out_shape_new.push_back(axis_c0);
  }

  if ((src_format == FORMAT_ND || src_format == FORMAT_NHWC || src_format == FORMAT_NCHW) &&
      dst_format == FORMAT_FRACTAL_NZ) {
    real_src_format = "HNC";
    real_dst_format = "HCNT";
    int64_t axis_n;
    int64_t axis_h;
    int64_t axis_c;
    if (in_shape.size() == 1) {
      axis_h = 1;
      axis_n = 1;
      axis_c = in_shape[0];
    } else if (in_shape.size() == 2) {
      axis_h = 1;
      axis_n = in_shape[0];
      axis_c = in_shape[1];
    } else {
      int64_t shape_size = 1;
      for (size_t i = 0; i < in_shape.size() - 2; i++) {
        shape_size *= in_shape[i];
      }
      axis_h = shape_size;
      axis_n = in_shape[in_shape.size() - 2];
      axis_c = in_shape[in_shape.size() - 1];
    }
    in_shape_new.push_back(axis_h);
    in_shape_new.push_back(axis_n);
    in_shape_new.push_back(axis_c);
    int64_t axis_c0 = c0_len;
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    int64_t axis_ni = NI_16;
    int64_t axis_no = GetCeilDiv(axis_n, axis_ni);
    out_shape_new.push_back(axis_h);
    out_shape_new.push_back(axis_c1);
    out_shape_new.push_back(axis_no * axis_ni);
    out_shape_new.push_back(axis_c0);
  }

  if (src_format == FORMAT_NC1HWC0 && dst_format == FORMAT_NCHW) {
    if (in_shape.size() < 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, in_shape size < 5");
      return false;
    }
    real_src_format = "NCHT";
    real_dst_format = "NCH";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[2] * in_shape[3]);
    in_shape_new.push_back(in_shape[4]);

    int64_t axis_c = out_shape[1];
    out_shape_new.push_back(in_shape[0]);
    out_shape_new.push_back(axis_c);
    out_shape_new.push_back(in_shape[2] * in_shape[3]);
  }

  if (src_format == FORMAT_NCDHW && dst_format == FORMAT_NDC1HWC0) {
    real_src_format = "NCDH";
    real_dst_format = "NDCHT";

    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[2]);
    in_shape_new.push_back(in_shape[3] * in_shape[4]);
    int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
    int64_t axis_c0 = c0_len;
    out_shape_new.push_back(in_shape[0]);
    out_shape_new.push_back(in_shape[2]);
    out_shape_new.push_back(axis_c1);
    out_shape_new.push_back(in_shape[3] * in_shape[4]);
    out_shape_new.push_back(axis_c0);
  }

  if (src_format == FORMAT_HWCN && dst_format == FORMAT_FRACTAL_Z) {
    real_src_format = "HCN";
    real_dst_format = "CHNT";

    if (in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    if (out_shape.size() < 2) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The output shape dimension size is not correct!");
      return false;
    }
    in_shape_new.push_back(in_shape[0] * in_shape[1]);
    in_shape_new.push_back(in_shape[2]);
    in_shape_new.push_back(in_shape[3]);

    int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axis_c0 = out_shape[out_shape.size() - 1];
    int64_t axis_ni = out_shape[out_shape.size() - 2];
    int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], axis_c0);
    int64_t axis_no = GetCeilDiv(in_shape[3], axis_ni);
    out_shape_new.push_back(axis_c1);
    out_shape_new.push_back(in_shape[0] * in_shape[1]);
    out_shape_new.push_back(axis_ni * axis_no);
    out_shape_new.push_back(axis_c0);
  }

  if (src_format == FORMAT_DHWCN && dst_format == FORMAT_FRACTAL_Z_3D) {
    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    real_src_format = "DHCN";
    real_dst_format = "DCHNT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1] * in_shape[2]);
    in_shape_new.push_back(in_shape[3]);
    in_shape_new.push_back(in_shape[4]);
    int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
    int64_t axis_c0 = c0_len;
    int64_t axis_ni = NI_16;
    int64_t axis_no = GetCeilDiv(in_shape[4], axis_ni);
    out_shape_new.push_back(in_shape[0]);
    out_shape_new.push_back(axis_c1);
    out_shape_new.push_back(in_shape[1] * in_shape[2]);
    out_shape_new.push_back(axis_no * axis_ni);
    out_shape_new.push_back(axis_c0);
  }

  if (src_format == FORMAT_NDHWC && dst_format == FORMAT_FRACTAL_Z_3D) {
    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    real_src_format = "NDHC";
    real_dst_format = "DCHNT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[3] * in_shape[2]);
    in_shape_new.push_back(in_shape[4]);
    int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
    int64_t axis_c0 = c0_len;
    int64_t axis_ni = NI_16;
    int64_t axis_no = GetCeilDiv(in_shape[0], axis_ni);
    out_shape_new.push_back(in_shape[1]);
    out_shape_new.push_back(axis_c1);
    out_shape_new.push_back(in_shape[3] * in_shape[2]);
    out_shape_new.push_back(axis_no * axis_ni);
    out_shape_new.push_back(axis_c0);
  }

  if (src_format == FORMAT_NC1HWC0 && dst_format == FORMAT_FRACTAL_Z) {
    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    real_src_format = "NDHC";
    real_dst_format = "DCHNT";
    int64_t axis_n = in_shape[0];
    int64_t axis_c1 = in_shape[1];
    int64_t axis_h = in_shape[2];
    int64_t axis_w = in_shape[3];
    int64_t axis_c0 = in_shape[4];
    int64_t axis_d = 1;
    int64_t axis_c = 1;
    int64_t axis_ni = NI_16;
    int64_t axis_no = GetCeilDiv(axis_n, axis_ni);
    in_shape_new = {axis_n, axis_d, axis_c1 * axis_h * axis_w, axis_c0};
    out_shape_new = {axis_d, axis_c, axis_c1 * axis_h * axis_w, axis_no * axis_ni, axis_c0};
  }

  if (src_format == FORMAT_NDHWC && dst_format == FORMAT_NDC1HWC0) {
    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    real_src_format = "NDHC";
    real_dst_format = "NDCHT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[3] * in_shape[2]);
    in_shape_new.push_back(in_shape[4]);
    int64_t c_idx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axis_c1 = GetCeilDiv(in_shape[c_idx], c0_len);
    int64_t axis_c0 = c0_len;
    out_shape_new.push_back(in_shape[0]);
    out_shape_new.push_back(in_shape[1]);
    out_shape_new.push_back(axis_c1);
    out_shape_new.push_back(in_shape[3] * in_shape[2]);
    out_shape_new.push_back(axis_c0);
  }

  if (src_format == FORMAT_NDC1HWC0 && dst_format == FORMAT_NCDHW) {
    if (in_shape.size() != 6 || out_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    real_src_format = "NDCHT";
    real_dst_format = "NCDH";
    int64_t axis_n = out_shape[0];
    int64_t axis_c = out_shape[1];
    int64_t axis_d = out_shape[2];
    int64_t axis_h = out_shape[3];
    int64_t axis_w = out_shape[4];
    int64_t axis_c0 = in_shape[in_shape.size() - 1];
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    in_shape_new = {axis_n, axis_d, axis_c1, axis_h * axis_w, axis_c0};
    out_shape_new = {axis_n, axis_c, axis_d, axis_h * axis_w};
  }

  if (src_format == FORMAT_FRACTAL_Z_3D && dst_format == FORMAT_NCDHW) {
    if (in_shape.size() < 2 || out_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    real_src_format = "DCHNT";
    real_dst_format = "NCDH";
    int64_t axis_n = out_shape[0];
    int64_t axis_c = out_shape[1];
    int64_t axis_d = out_shape[2];
    int64_t axis_h = out_shape[3];
    int64_t axis_w = out_shape[4];
    int64_t axis_c0 = in_shape[in_shape.size() - 1];
    int64_t axis_ni = in_shape[in_shape.size() - 2];
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    int64_t axis_no = GetCeilDiv(axis_n, axis_ni);
    in_shape_new = {axis_d, axis_c1, axis_h * axis_w, axis_no * axis_ni, axis_c0};
    out_shape_new = {axis_n, axis_c, axis_d, axis_h * axis_w};
  }

  if (src_format == FORMAT_NC1HWC0 && dst_format == FORMAT_NHWC) {
    if (in_shape.size() != 5 || out_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data",
                                      "The input shape dimension size should be 5 and output's should be 4!");
      return false;
    }
    real_src_format = "NCHT";
    real_dst_format = "NHC";
    int64_t axis_n = in_shape[0];
    int64_t axis_c1 = in_shape[1];
    int64_t axis_h = in_shape[2];
    int64_t axis_w = in_shape[3];
    int64_t axis_c0 = in_shape[4];
    int64_t axis_c = out_shape[out_shape.size() - 1];
    int64_t axis_hw = axis_h * axis_w;
    in_shape_new = {axis_n, axis_c1, axis_hw, axis_c0};
    out_shape_new = {axis_n, axis_hw, axis_c};
  }

  if (src_format == FORMAT_NDC1HWC0 && dst_format == FORMAT_NDHWC) {
    if (in_shape.size() != 6 || out_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data",
                                      "The input shape dimension size should be 6 and output's should be 5!");
      return false;
    }
    real_src_format = "NCHT";
    real_dst_format = "NHC";
    int64_t axis_n = in_shape[0];
    int64_t axis_d = in_shape[1];
    int64_t axis_c1 = in_shape[2];
    int64_t axis_h = in_shape[3];
    int64_t axis_w = in_shape[4];
    int64_t axis_c0 = in_shape[5];
    int64_t axis_c = out_shape[out_shape.size() - 1];
    int64_t axis_hw = axis_h * axis_w;
    in_shape_new = {axis_n * axis_d, axis_c1, axis_hw, axis_c0};
    out_shape_new = {axis_n * axis_d, axis_hw, axis_c};
  }

  if ((src_format == FORMAT_FRACTAL_NZ) &&
      (dst_format == FORMAT_ND || dst_format == FORMAT_NCHW || dst_format == FORMAT_NHWC)) {
    if (out_shape.size() == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The output shape dimension size cannot be 0!");
      return false;
    }
    real_src_format = "HCNT";
    real_dst_format = "HNC";
    int64_t axis_n;
    int64_t axis_h;
    int64_t axis_c;
    if (out_shape.size() == 1) {
      axis_h = 1;
      axis_n = 1;
      axis_c = out_shape[0];
    } else if (out_shape.size() == 2) {
      axis_h = 1;
      axis_n = out_shape[0];
      axis_c = out_shape[1];
    } else {
      int64_t shape_size = 1;
      for (size_t i = 0; i < out_shape.size() - 2; i++) {
        shape_size *= out_shape[i];
      }
      axis_h = shape_size;
      axis_n = out_shape[out_shape.size() - 2];
      axis_c = out_shape[out_shape.size() - 1];
    }
    int64_t axis_c0 = in_shape[in_shape.size() - 1];
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    int64_t axis_no = GetCeilDiv(axis_n, NI_16);
    in_shape_new = {axis_h, axis_c1, axis_no * NI_16, axis_c0};
    out_shape_new = {axis_h, axis_n, axis_c};
  }

  if (src_format == FORMAT_FRACTAL_Z_3D && dst_format == FORMAT_NDHWC) {
    if (in_shape.size() != 4 || out_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    real_src_format = "DCHNT";
    real_dst_format = "NDHC";
    int64_t axis_dc1hw = in_shape[0];
    int64_t axis_no = in_shape[1];
    int64_t axis_ni = in_shape[2];
    int64_t axis_c0 = in_shape[3];
    int64_t axis_n = out_shape[0];
    int64_t axis_d = out_shape[1];
    int64_t axis_h = out_shape[2];
    int64_t axis_w = out_shape[3];
    int64_t axis_c = out_shape[4];
    int64_t axis_c1 = axis_dc1hw / (axis_d * axis_h * axis_w);

    in_shape_new = {axis_d, axis_c1, axis_h * axis_w, axis_no * axis_ni, axis_c0};
    out_shape_new = {axis_n, axis_d, axis_h * axis_w, axis_c};
  }

  if (src_format == FORMAT_FRACTAL_NZ && dst_format == FORMAT_NC1HWC0) {
    if (out_shape.size() != 5 || in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    real_src_format = "DCHNT";
    real_dst_format = "NDHC";

    int64_t axis_d = 1;
    int64_t axis_c = 1;
    int64_t axis_n = out_shape[0];
    int64_t axis_c1 = out_shape[1];
    int64_t axis_h = out_shape[2];
    int64_t axis_w = out_shape[3];
    int64_t axis_c0 = out_shape[4];
    int64_t axis_no = GetCeilDiv(axis_n, NI_16);
    int64_t axis_ni = NI_16;

    in_shape_new = {axis_d, axis_c, axis_c1 * axis_h * axis_w, axis_no * axis_ni, axis_c0};
    out_shape_new = {axis_n, axis_d, axis_c1 * axis_h * axis_w, axis_c0};
  }

  if (src_format == FORMAT_FRACTAL_Z && dst_format == FORMAT_HWCN) {
    real_src_format = "CHNT";
    real_dst_format = "HCN";

    if (out_shape.size() != 4 || in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect!");
      return false;
    }

    int64_t axis_h = out_shape[0];
    int64_t axis_w = out_shape[1];
    int64_t axis_c = out_shape[2];
    int64_t axis_n = out_shape[3];
    int64_t axis_c0 = in_shape[in_shape.size() - 1];
    int64_t axis_ni = in_shape[in_shape.size() - 2];
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    int64_t axis_no = GetCeilDiv(axis_n, axis_ni);

    in_shape_new = {axis_c1, axis_h * axis_w, axis_no * axis_ni, axis_c0};
    out_shape_new = {axis_h * axis_w, axis_c, axis_n};
  }

  if (src_format == FORMAT_FRACTAL_Z_3D && dst_format == FORMAT_DHWCN) {
    if (in_shape.size() != 4 || out_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    real_src_format = "DCHNT";
    real_dst_format = "DHCN";
    int64_t axis_d = out_shape[0];
    int64_t axis_h = out_shape[1];
    int64_t axis_w = out_shape[2];
    int64_t axis_c = out_shape[3];
    int64_t axis_n = out_shape[4];
    int64_t axis_c0 = in_shape[in_shape.size() - 1];
    int64_t axis_ni = in_shape[in_shape.size() - 2];
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    int64_t axis_no = GetCeilDiv(axis_n, axis_ni);
    in_shape_new = {axis_d, axis_c1, axis_h * axis_w, axis_no * axis_ni, axis_c0};
    out_shape_new = {axis_d, axis_h * axis_w, axis_c, axis_n};
  }

  if (src_format == FORMAT_FRACTAL_Z && dst_format == FORMAT_NCHW) {
    real_src_format = "CHNT";
    real_dst_format = "NCH";

    if (out_shape.size() != 4 || in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect!");
      return false;
    }

    int64_t axis_n = out_shape[0];
    int64_t axis_c = out_shape[1];
    int64_t axis_h = out_shape[2];
    int64_t axis_w = out_shape[3];
    int64_t axis_c0 = in_shape[in_shape.size() - 1];
    int64_t axis_ni = in_shape[in_shape.size() - 2];
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    int64_t axis_no = GetCeilDiv(axis_n, axis_ni);

    in_shape_new = {axis_c1, axis_h * axis_w, axis_no * axis_ni, axis_c0};
    out_shape_new = {axis_n, axis_c, axis_h * axis_w};
  }

  if (src_format == FORMAT_FRACTAL_Z && dst_format == FORMAT_ND) {
    real_src_format = "HCNT";
    real_dst_format = "HCN";

    if (in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect!");
      return false;
    }
    int64_t axis_n;
    int64_t axis_h;
    int64_t axis_c;
    if (out_shape.size() == 1) {
      axis_h = 1;
      axis_c = 1;
      axis_n = out_shape[0];
    } else if (out_shape.size() == 2) {
      axis_h = 1;
      axis_c = out_shape[0];
      axis_n = out_shape[1];
    } else {
      int64_t shape_size = 1;
      for (size_t i = 0; i < out_shape.size() - 2; i++) {
        shape_size *= out_shape[i];
      }
      axis_h = shape_size;
      axis_c = out_shape[out_shape.size() - 2];
      axis_n = out_shape[out_shape.size() - 1];
    }
    int64_t axis_c0 = in_shape[in_shape.size() - 1];
    int64_t axis_ni = in_shape[in_shape.size() - 2];
    int64_t axis_c1 = GetCeilDiv(axis_c, axis_c0);
    int64_t axis_no = GetCeilDiv(axis_n, axis_ni);

    in_shape_new = {axis_h, axis_c1, axis_no * axis_ni, axis_c0};
    out_shape_new = {axis_h, axis_c, axis_n};
  }

  return true;
}

int32_t GetMultiCoreAxis(std::vector<int64_t> in_shape, int32_t axis_pos_c, int64_t block_elem_cnt, int64_t c0_len,
                         int64_t core_num) {
  int32_t shape_len = in_shape.size();
  bool axis_c_not_last_dim = axis_pos_c + 1 != shape_len;
  std::vector<int32_t> core_lp_cnt;

  for (int32_t index = 0; index < shape_len; index++) {
    int32_t tmp_full_cycle_loop_cnt;
    int32_t left_loop_cnt;
    int32_t full_cycle_loop_cnt;
    if (index + 1 == shape_len) {
      if (GetFloorDiv(in_shape[index], 8 * block_elem_cnt * core_num) > 0) {
        tmp_full_cycle_loop_cnt = core_num;
      } else {
        tmp_full_cycle_loop_cnt = 0;
      }
      left_loop_cnt = GetCeilDiv(in_shape[index], 8 * block_elem_cnt) % core_num;
    } else if (index == axis_pos_c && axis_c_not_last_dim) {
      if (GetFloorDiv(in_shape[index], c0_len * core_num) > 0) {
        tmp_full_cycle_loop_cnt = core_num;
      } else {
        tmp_full_cycle_loop_cnt = 0;
      }
      left_loop_cnt = GetCeilDiv(in_shape[index], c0_len) % core_num;
    } else {
      if (GetFloorDiv(in_shape[index], core_num) > 0) {
        tmp_full_cycle_loop_cnt = core_num;
      } else {
        tmp_full_cycle_loop_cnt = 0;
      }
      left_loop_cnt = in_shape[index] % core_num;
    }

    if (tmp_full_cycle_loop_cnt > 0 && left_loop_cnt == 0) {
      full_cycle_loop_cnt = 2 * tmp_full_cycle_loop_cnt;
    } else {
      full_cycle_loop_cnt = tmp_full_cycle_loop_cnt;
    }
    core_lp_cnt.push_back(full_cycle_loop_cnt + left_loop_cnt);
  }

  return max_element(core_lp_cnt.begin(), core_lp_cnt.end()) - core_lp_cnt.begin();
}

bool IsDoWithTransposeFormats(const ge::Format& src_format, const ge::Format& dst_format) {
  const std::vector<ge::Format> format_list = {FORMAT_NCHW, FORMAT_NHWC, FORMAT_HWCN, FORMAT_CHWN};
  if (std::find(format_list.begin(), format_list.end(), src_format) != format_list.end() &&
      std::find(format_list.begin(), format_list.end(), dst_format) != format_list.end() && dst_format != src_format) {
    return true;
  } else {
    return false;
  }
}

bool IsDoWithPositiveSourceNtc100(const ge::Format& src_format, const ge::Format& dst_format) {
  const std::vector<std::pair<ge::Format, ge::Format>> support_src_dst_formats = {
    {FORMAT_NCDHW, FORMAT_NDC1HWC0}, {FORMAT_NCHW, FORMAT_NC1HWC0}, {FORMAT_HWCN, FORMAT_FRACTAL_Z},
    {FORMAT_DHWCN, FORMAT_FRACTAL_Z_3D}, {FORMAT_NCDHW, FORMAT_FRACTAL_Z_3D}, {FORMAT_ND, FORMAT_FRACTAL_Z},
    {FORMAT_NCHW, FORMAT_FRACTAL_Z}
  };
  return std::find(support_src_dst_formats.begin(), support_src_dst_formats.end(),
                   std::pair<ge::Format, ge::Format>(src_format, dst_format)) != support_src_dst_formats.end();
}

bool IsDoWithNegativeTargetTc201(const ge::Format& src_format, const ge::Format& dst_format) {
  const std::vector<std::pair<ge::Format, ge::Format>> support_src_dst_formats = {
    {FORMAT_NC1HWC0, FORMAT_NHWC}, {FORMAT_FRACTAL_NZ, FORMAT_ND}, {FORMAT_FRACTAL_NZ, FORMAT_NCHW},
    {FORMAT_FRACTAL_NZ, FORMAT_NHWC}, {FORMAT_FRACTAL_Z_3D, FORMAT_NDHWC}, {FORMAT_FRACTAL_NZ, FORMAT_NC1HWC0},
    {FORMAT_NDC1HWC0, FORMAT_NDHWC}
  };
  return std::find(support_src_dst_formats.begin(), support_src_dst_formats.end(),
                   std::pair<ge::Format, ge::Format>(src_format, dst_format)) != support_src_dst_formats.end();
}

bool IsDoWithNegativeTargetNtc200(const ge::Format& src_format, const ge::Format& dst_format) {
  const std::vector<std::pair<ge::Format, ge::Format>> support_src_dst_formats = {
    {FORMAT_NC1HWC0, FORMAT_NCHW}, {FORMAT_FRACTAL_Z, FORMAT_HWCN}, {FORMAT_FRACTAL_Z, FORMAT_NCHW},
    {FORMAT_FRACTAL_Z, FORMAT_ND}, {FORMAT_FRACTAL_Z_3D, FORMAT_NCDHW}, {FORMAT_FRACTAL_Z_3D, FORMAT_DHWCN},
    {FORMAT_NDC1HWC0, FORMAT_NCDHW}
  };
  return std::find(support_src_dst_formats.begin(), support_src_dst_formats.end(),
                   std::pair<ge::Format, ge::Format>(src_format, dst_format)) != support_src_dst_formats.end();
}

/*
 * @brief: tiling function of op
 * @param [in] op_type: op_type of the op
 * @param [in] op_paras: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool TransDataTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                     utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
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

  auto output_desc = operator_info->MutableOutputDesc(0);
  if (output_desc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get output desc failed.");
    return false;
  }

  ge::Format src_format = input_desc->GetFormat();
  ge::Format dst_format = output_desc->GetFormat();
  OP_LOGD(op_type, "Input format is [%s], Output format is [%s].", to_string(src_format), to_string(dst_format));
  std::vector<int64_t> in_shape = input_desc->MutableShape().GetDims();
  std::vector<int64_t> out_shape = output_desc->MutableShape().GetDims();
  auto data_type = input_desc->GetDataType();
  // get the point ts after get info from op_paras
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  std::string real_src_format;
  std::string real_dst_format;
  int64_t ub_size = 0;
  int64_t block_dim = 0;
  std::vector<int64_t> in_shape_new;
  std::vector<int64_t> out_shape_new;
  in_shape_new.reserve(8);
  out_shape_new.reserve(8);
  int64_t group = 1;
  int64_t vnc_fp32_flag = 0;

  bool flag = GetCompileParams(op_info, ub_size, block_dim, group, vnc_fp32_flag, op_type);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileParams error.");
    return false;
  }

  // get the point ts after get compile info
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  int64_t c0_len = GetC0SizeWithType(data_type);
  bool ret = CheckTensorShape(op_type, ub_size, block_dim, out_shape);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CheckTensor Failed.");
    return ret;
  }
  int64_t block_elem_cnt = BLOCK_BYTE_SIZE / GetSizeByDataType(data_type);

  flag = GetRenew2Shape(in_shape, out_shape, src_format, dst_format, c0_len, group, in_shape_new,
                        out_shape_new, real_src_format, real_dst_format);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetRenew2Shape tiling params error");
    return false;
  }

  if (IsDoWithPositiveSourceNtc100(src_format, dst_format)) {
    TransDataNtc100Param run_params_100;
    flag = TilingPositiveSourceNtc100(in_shape, out_shape, src_format, dst_format, block_dim,
                                      block_elem_cnt, ub_size, c0_len, data_type, run_params_100);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get TilingPositiveSourceNtc100 tiling params error");
      return false;
    }
    PROFILING_TILING_AFTER_CALCU_TILING_REG();
    SetRunningNtc100Params(run_params_100, run_info);
    OP_LOGD(op_type, "start print tiling parameters in ntc 100: %s", run_params_100.to_string().c_str());
  } else if (real_src_format[real_src_format.length() - 1] == 'C' && real_dst_format[real_dst_format.length() - 1] == 'T') {
    if (real_src_format[real_src_format.length() - 2] == real_dst_format[real_dst_format.length() - 2]) {
      TransDataMode1010Param run_params_part1;
      flag = TillingPositiveMode1010(in_shape_new, out_shape_new, real_src_format, real_dst_format,
                                     block_dim, block_elem_cnt, ub_size, run_params_part1);
      if (!flag) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get TransDataMode101Param tiling params error");
        return false;
      }
      PROFILING_TILING_AFTER_CALCU_TILING_REG();
      SetRunningMode1010Params(run_params_part1, run_info);
      OP_LOGD(op_type, "start print runParams");
      PrintTilingMode1010Params(op_type, run_params_part1);
    } else {
      TransDataMode1011Param run_params_part1;
      flag = TillingPositiveMode1011(in_shape_new, out_shape_new, real_src_format, real_dst_format,
                                     block_dim, block_elem_cnt, ub_size, run_params_part1);
      if (!flag) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get TransDataMode101Param tiling params error");
        return false;
      }
      PROFILING_TILING_AFTER_CALCU_TILING_REG();
      SetRunningMode1011Params(run_params_part1, run_info);
      OP_LOGD(op_type, "start print runParams");
      PrintTilingMode1011Params(op_type, run_params_part1);
    }

  } else if (IsDoWithNegativeTargetTc201(src_format, dst_format)) {
    TransDataTc201Param run_params_201;
    flag = TilingNegativeTc201(in_shape_new, out_shape_new, real_src_format, real_dst_format, block_dim,
                               block_elem_cnt, data_type, ub_size, run_params_201);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get TilingNegativeTc201 tiling params error");
      return false;
    }
    OP_LOGD(op_type, "***start to put mode 201 tiling parameters");
    PROFILING_TILING_AFTER_CALCU_TILING_REG();
    SetRunningTc201Params(run_params_201, run_info);
    PrintTilingModeTc201Params(op_type, run_params_201);
  } else if (IsDoWithNegativeTargetNtc200(src_format, dst_format)) {
    TransDataNtc200Param run_params_200;
    flag = TilingNegativeNtc200(in_shape_new, out_shape_new, real_src_format, real_dst_format, block_dim,
                                block_elem_cnt, data_type, ub_size, vnc_fp32_flag, run_params_200);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get TilingNegativeNtc200 tiling params error");
      return false;
    }
    PROFILING_TILING_AFTER_CALCU_TILING_REG();
    SetRunningNtc200Params(run_params_200, run_info);
    OP_LOGD(op_type, "start print tiling parameters in mode 200");
    PrintTilingModeNtc200Params(op_type, run_params_200);
  }

  // block_dim, core num used in tik op
  run_info.SetBlockDim(block_dim);

  OP_LOGI(op_type, "tiling run success.");
  // get the point ts and calcu the all time cost
  PROFILING_TILING_END();
  return true;
}

// register tiling interface of the TransData op
REGISTER_OP_TILING_FUNC_BUFFERED_V2(TransData, TransDataTiling);

}  // namespace optiling
