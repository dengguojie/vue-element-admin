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
#include "trans_data.h"

using namespace gert;

namespace optiling {
namespace transdata {
int64_t GetC0SizeWithType(ge::DataType& dtype) {
  if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
    return C0_32;
  }
  return C0_16;
}

int64_t GetShapeSize(const gert::Shape& in_shape, int32_t pos) {
  int32_t n = in_shape.GetDimNum();
  int64_t shape_size = 1;
  if (pos < 0) {
    pos = n + pos;
  }
  for (int32_t i = pos; i < n; i++) {
    shape_size *= in_shape[i];
  }
  return shape_size;
}

std::string RealFormatToSerialString(RealFormat format) {
  auto it = RealFormatToStringMap.find(format);
  if (it != RealFormatToStringMap.end()) {
    return it->second;
  } else {
    return "RESERVED";
  }
}

bool ConvertShapeHNC2HCNT(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                          gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  int64_t axis_n;
  int64_t axis_h;
  int64_t axis_c;
  if (in_shape.GetDimNum() == 1) {
    axis_h = 1;
    axis_n = 1;
    axis_c = in_shape[0];
  } else if (in_shape.GetDimNum() == DIM_NUM_2) {
    axis_h = 1;
    axis_n = in_shape[0];
    axis_c = in_shape[1];
  } else {
    int64_t shape_size = 1;
    for (size_t i = 0; i < in_shape.GetDimNum() - DIM_NUM_2; i++) {
      shape_size *= in_shape[i];
    }
    axis_h = shape_size;
    axis_n = in_shape[in_shape.GetDimNum() + DIM_IDX_NEG_TWO];
    axis_c = in_shape[in_shape.GetDimNum() + DIM_IDX_NEG_ONE];
  }
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_n);
  in_shape_new.AppendDim(axis_c);
  int64_t axis_c0 = c0_size;
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_ni = NI_16;
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(axis_no * axis_ni);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool Convert5HDToNCHW(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() < SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Convert5HDToNCHW failed, in_shape size < 5");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_4]);

  int64_t axis_c = out_shape[1];
  out_shape_new.AppendDim(in_shape[0]);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  return true;
}

bool Convert5HDToNHWC(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D || out_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        "TransData", "Convert5HDToNHWC failed, The input shape dimension size should be 5 and output's should be 4!");
    return false;
  }
  int64_t axis_n = in_shape[0];
  int64_t axis_c1 = in_shape[1];
  int64_t axis_h = in_shape[DIM_IDX_2];
  int64_t axis_w = in_shape[DIM_IDX_3];
  int64_t axis_c0 = in_shape[DIM_IDX_4];
  int64_t axis_c = out_shape[out_shape.GetDimNum() - 1];
  int64_t axis_hw = axis_h * axis_w;
  in_shape_new.AppendDim(axis_n);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_hw);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_hw);
  out_shape_new.AppendDim(axis_c);
  return true;
}

bool ConvertShapeHCNT2HNC(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                          gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  int64_t axis_n;
  int64_t axis_h;
  int64_t axis_c;
  size_t out_shape_size = out_shape.GetDimNum();
  if (out_shape_size == 1) {
    axis_h = 1;
    axis_n = 1;
    axis_c = out_shape[0];
  } else if (out_shape_size == DIM_NUM_2) {
    axis_h = 1;
    axis_n = out_shape[0];
    axis_c = out_shape[1];
  } else {
    int64_t shape_size = 1;
    for (size_t i = 0; i < out_shape_size - DIM_NUM_2; i++) {
      shape_size *= out_shape[i];
    }
    axis_h = shape_size;
    axis_n = out_shape[out_shape_size + DIM_IDX_NEG_TWO];
    axis_c = out_shape[out_shape_size + DIM_IDX_NEG_ONE];
  }
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_ni = NI_16;
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c);
  return true;
}

bool ConvertNCHWTo5HD(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != transdata::SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData",
                                    "ConvertNCHWTo5HD failed, The input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  int64_t c_idx = 1;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  out_shape_new.AppendDim(in_shape[0]);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  out_shape_new.AppendDim(c0_size);
  return true;
}

bool ConvertNHWCTo5HD(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  static const int64_t HW_IDX = 1;
  static const int64_t C_IDX = 3;
  if (in_shape.GetDimNum() < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertNHWCTo5HD failed, in_shape size < 1");
    return false;
  }
  for (int64_t i = 0; i < HW_IDX; i++) {
    in_shape_new.AppendDim(in_shape[i]);
  }
  int64_t n = in_shape.GetDimNum() - 1;
  int64_t shape_size = 1;
  for (int64_t i = HW_IDX; i < n; i++) {
    shape_size *= in_shape[i];
  }
  in_shape_new.AppendDim(shape_size);
  in_shape_new.AppendDim(in_shape[in_shape.GetDimNum() - 1]);
  int64_t axis_c1 = ge::CeilDiv(in_shape[C_IDX], c0_size);
  int64_t axis_n = in_shape[0];
  int64_t axis_h = in_shape_new[HW_IDX];
  int64_t axis_c0 = c0_size;
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool ConvertNCDHWTo6HD(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                       gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData",
                                    "ConvertNCDHWTo6HD failed, The input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_4]);
  int64_t c_idx = 1;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  int64_t axis_c0 = c0_size;
  out_shape_new.AppendDim(in_shape[0]);
  out_shape_new.AppendDim(in_shape[DIM_IDX_2]);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_4]);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool ConvertNDHWCTo6HD(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                       gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData",
                                    "ConvertNDHWCTo6HD failed, The input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_2]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_4]);
  int64_t c_idx = DIM_IDX_4;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  int64_t axis_c0 = c0_size;
  out_shape_new.AppendDim(in_shape[0]);
  out_shape_new.AppendDim(in_shape[1]);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_2]);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool Convert6HDToNCDHW(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                       gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_6D || out_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Convert6HDToNCDHW failed, shape size incorrect");
    return false;
  }
  int64_t axis_n = out_shape[0];
  int64_t axis_c = out_shape[1];
  int64_t axis_d = out_shape[DIM_IDX_2];
  int64_t axis_h = out_shape[DIM_IDX_3];
  int64_t axis_w = out_shape[DIM_IDX_4];
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  in_shape_new.AppendDim(axis_n);
  in_shape_new.AppendDim(axis_d);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_h * axis_w);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(axis_d);
  out_shape_new.AppendDim(axis_h * axis_w);
  return true;
}

bool Convert6HDToNDHWC(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                       gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_6D || out_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Convert6HDToNDHWC failed, shape size incorrect");
    return false;
  }
  int64_t axis_n = in_shape[0];
  int64_t axis_d = in_shape[1];
  int64_t axis_c1 = in_shape[DIM_IDX_2];
  int64_t axis_h = in_shape[DIM_IDX_3];
  int64_t axis_w = in_shape[DIM_IDX_4];
  int64_t axis_c0 = in_shape[DIM_IDX_5];
  int64_t axis_c = out_shape[out_shape.GetDimNum() - 1];
  int64_t axis_hw = axis_h * axis_w;
  in_shape_new.AppendDim(axis_n * axis_d);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_hw);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_n * axis_d);
  out_shape_new.AppendDim(axis_hw);
  out_shape_new.AppendDim(axis_c);
  return true;
}

bool ConvertHWCNToFZ(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                     gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertHWCNToFZ failed, input shape dimension size is not correct!");
    return false;
  }
  if (out_shape.GetDimNum() < SHAPE_LEN_2D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertHWCNToFZ failed, output shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0] * in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_3]);

  int64_t c_idx = DIM_IDX_2;
  int64_t axis_c0 = out_shape[out_shape.GetDimNum() - 1];
  int64_t axis_ni = out_shape[out_shape.GetDimNum() - DIM_IDX_2];
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], axis_c0);
  int64_t axis_no = ge::CeilDiv(in_shape[DIM_IDX_3], axis_ni);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[0] * in_shape[1]);
  out_shape_new.AppendDim(axis_ni * axis_no);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool ConvertDHWCNToFZ3D(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                        gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData",
                                    "ConvertDHWCNToFZ3D failed,input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1] * in_shape[DIM_IDX_2]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_3]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_4]);
  int64_t c_idx = DIM_IDX_3;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  int64_t axis_c0 = c0_size;
  int64_t axis_ni = NI_16;
  int64_t axis_no = ge::CeilDiv(in_shape[DIM_IDX_4], axis_ni);
  out_shape_new.AppendDim(in_shape[0]);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[1] * in_shape[DIM_IDX_2]);
  out_shape_new.AppendDim(axis_no * axis_ni);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool ConvertNDHWCToFZ3D(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                        gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData",
                                    "ConvertNDHWCToFZ3D failed, input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_2]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_4]);
  int64_t c_idx = DIM_IDX_4;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  int64_t axis_c0 = c0_size;
  int64_t axis_ni = NI_16;
  int64_t axis_no = ge::CeilDiv(in_shape[0], axis_ni);
  out_shape_new.AppendDim(in_shape[1]);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_2]);
  out_shape_new.AppendDim(axis_no * axis_ni);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool ConvertNCDHWToFZ3D(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                        gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData",
                                    "ConvertNCDHWToFZ3D failed, input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_4]);
  int64_t c_idx = 1;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  int64_t n_idx = 0;
  int64_t axis_no = ge::CeilDiv(in_shape[n_idx], NI_16);
  out_shape_new.AppendDim(in_shape[DIM_IDX_2]);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[DIM_IDX_3] * in_shape[DIM_IDX_4]);
  out_shape_new.AppendDim(NI_16 * axis_no);
  out_shape_new.AppendDim(c0_size);
  return true;
}

bool ConvertNDToFZ(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                   gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  int64_t axis_h = 1;
  int64_t axis_c = 1;
  int64_t axis_n = 1;
  size_t in_shape_dim_size = in_shape.GetDimNum();
  if (in_shape_dim_size == 1) {
    axis_h = 1;
    axis_c = 1;
    axis_n = in_shape[0];
  } else if (in_shape_dim_size == SHAPE_LEN_2D) {
    axis_h = 1;
    axis_c = in_shape[0];
    axis_n = in_shape[1];
  } else {
    for (size_t i = 0; i < in_shape_dim_size - DIM_NUM_2; i++) {
      axis_h *= in_shape[i];
    }
    axis_c = in_shape[in_shape_dim_size - DIM_NUM_2];
    axis_n = in_shape[in_shape_dim_size - 1];
  }
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_c);
  in_shape_new.AppendDim(axis_n);
  int64_t axis_c1 = ge::CeilDiv(axis_c, c0_size);
  int64_t axis_no = ge::CeilDiv(axis_n, NI_16);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(axis_no * NI_16);
  out_shape_new.AppendDim(c0_size);
  return true;
}

bool ConvertNCHWToFZ(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                     gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertNCHWToFZ failed, input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  int64_t c_idx = 1;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  int64_t n_idx = 0;
  int64_t axis_no = ge::CeilDiv(in_shape[n_idx], NI_16);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  out_shape_new.AppendDim(NI_16 * axis_no);
  out_shape_new.AppendDim(c0_size);
  return true;
}

bool Convert5HDToFZ(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                    gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Convert5HDToFZ failed, input shape dimension size is not correct!");
    return false;
  }
  int64_t axis_n = in_shape[0];
  int64_t axis_c1 = in_shape[1];
  int64_t axis_h = in_shape[DIM_IDX_2];
  int64_t axis_w = in_shape[DIM_IDX_3];
  int64_t axis_c0 = in_shape[DIM_IDX_4];
  int64_t axis_d = 1;
  int64_t axis_c = 1;
  int64_t axis_ni = NI_16;
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_n);
  in_shape_new.AppendDim(axis_d);
  in_shape_new.AppendDim(axis_c1 * axis_h * axis_w);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_d);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(axis_c1 * axis_h * axis_w);
  out_shape_new.AppendDim(axis_no * axis_ni);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool ConvertFZ3DToNCDHW(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                        gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() < SHAPE_LEN_2D || out_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertFZ3DToNCDHW failed, shape size incorrect");
    return false;
  }
  int64_t axis_n = out_shape[0];
  int64_t axis_c = out_shape[1];
  int64_t axis_d = out_shape[DIM_IDX_2];
  int64_t axis_h = out_shape[DIM_IDX_3];
  int64_t axis_w = out_shape[DIM_IDX_4];
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_ni = in_shape[in_shape.GetDimNum() - DIM_NUM_2];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_d);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_h * axis_w);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(axis_d);
  out_shape_new.AppendDim(axis_h * axis_w);
  return true;
}

bool ConvertFZ3DToNDHWC(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                        gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_4D || out_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertFZ3DToNDHWC failed, shape size incorrect");
    return false;
  }
  int64_t axis_dc1hw = in_shape[0];
  int64_t axis_no = in_shape[1];
  int64_t axis_ni = in_shape[DIM_IDX_2];
  int64_t axis_c0 = in_shape[DIM_IDX_3];
  int64_t axis_n = out_shape[0];
  int64_t axis_d = out_shape[1];
  int64_t axis_h = out_shape[DIM_IDX_2];
  int64_t axis_w = out_shape[DIM_IDX_3];
  int64_t axis_c = out_shape[DIM_IDX_4];
  int64_t axis_c1 = ge::FloorDiv(axis_dc1hw, axis_d * axis_h * axis_w);
  in_shape_new.AppendDim(axis_d);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_h * axis_w);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_d);
  out_shape_new.AppendDim(axis_h * axis_w);
  out_shape_new.AppendDim(axis_c);
  return true;
}

bool ConvertNZTo5HD(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                    gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (out_shape.GetDimNum() != SHAPE_LEN_5D || in_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertNZTo5HD failed, shape size incorrect");
    return false;
  }
  int64_t axis_d = 1;
  int64_t axis_c = 1;
  int64_t axis_n = out_shape[0];
  int64_t axis_c1 = out_shape[1];
  int64_t axis_h = out_shape[DIM_IDX_2];
  int64_t axis_w = out_shape[DIM_IDX_3];
  int64_t axis_c0 = out_shape[DIM_IDX_4];
  int64_t axis_no = ge::CeilDiv(axis_n, NI_16);
  int64_t axis_ni = NI_16;
  in_shape_new.AppendDim(axis_d);
  in_shape_new.AppendDim(axis_c);
  in_shape_new.AppendDim(axis_c1 * axis_h * axis_w);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_d);
  out_shape_new.AppendDim(axis_c1 * axis_h * axis_w);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool ConvertFZ3DToDHWCN(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                        gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_4D || out_shape.GetDimNum() != SHAPE_LEN_5D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertFZ3DToDHWCN failed, shape size incorrect");
    return false;
  }
  int64_t axis_d = out_shape[0];
  int64_t axis_h = out_shape[1];
  int64_t axis_w = out_shape[DIM_IDX_2];
  int64_t axis_c = out_shape[DIM_IDX_3];
  int64_t axis_n = out_shape[DIM_IDX_4];
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_ni = in_shape[in_shape.GetDimNum() - DIM_NUM_2];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_d);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_h * axis_w);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_d);
  out_shape_new.AppendDim(axis_h * axis_w);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(axis_n);
  return true;
}

bool ConvertFZToHWCN(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                     gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (out_shape.GetDimNum() != SHAPE_LEN_4D || in_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertFZToHWCN failed, shape size incorrect!");
    return false;
  }
  int64_t axis_h = out_shape[0];
  int64_t axis_w = out_shape[1];
  int64_t axis_c = out_shape[2];
  int64_t axis_n = out_shape[3];
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_ni = in_shape[in_shape.GetDimNum() - DIM_NUM_2];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_h * axis_w);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_h * axis_w);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(axis_n);
  return true;
}

bool ConvertFZToNCHW(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                     gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (out_shape.GetDimNum() != SHAPE_LEN_4D || in_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertFZToNCHW failed, shape size incorrect!");
    return false;
  }
  int64_t axis_n = out_shape[0];
  int64_t axis_c = out_shape[1];
  int64_t axis_h = out_shape[DIM_IDX_2];
  int64_t axis_w = out_shape[DIM_IDX_3];
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_ni = in_shape[in_shape.GetDimNum() - DIM_NUM_2];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_h * axis_w);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(axis_h * axis_w);
  return true;
}

bool ConvertFZToND(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                   gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertFZToND failed, shape size incorrect!");
    return false;
  }
  int64_t axis_n;
  int64_t axis_h;
  int64_t axis_c;
  if (out_shape.GetDimNum() == 1) {
    axis_h = 1;
    axis_c = 1;
    axis_n = out_shape[0];
  } else if (out_shape.GetDimNum() == SHAPE_LEN_2D) {
    axis_h = 1;
    axis_c = out_shape[0];
    axis_n = out_shape[1];
  } else {
    int64_t shape_size = 1;
    for (size_t i = 0; i < out_shape.GetDimNum() - DIM_NUM_2; i++) {
      shape_size *= out_shape[i];
    }
    axis_h = shape_size;
    axis_c = out_shape[out_shape.GetDimNum() - DIM_NUM_2];
    axis_n = out_shape[out_shape.GetDimNum() - 1];
  }
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_ni = in_shape[in_shape.GetDimNum() - DIM_NUM_2];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(axis_n);
  return true;
}

size_t format_nd_array[ARRAYNDLEN] = {ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC};
size_t format_4d_array[ARRAY4DLEN] = {ge::FORMAT_NCHW, ge::FORMAT_NHWC};
RealShapeConvertFunc GetRealShapeConvertFunc(ge::Format src_format, ge::Format dst_format) {
  static auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, RealShapeConvertFunc>(nullptr)
                          .Add(format_nd_array, ARRAYNDLEN, ge::FORMAT_FRACTAL_NZ, ConvertShapeHNC2HCNT)
                          .Add(ge::FORMAT_FRACTAL_NZ, format_nd_array, ARRAYNDLEN, ConvertShapeHCNT2HNC)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ConvertNCHWTo5HD)
                          .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ConvertNHWCTo5HD)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, Convert5HDToNCHW)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC, Convert5HDToNHWC)
                          .Add(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ConvertNCDHWTo6HD)
                          .Add(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, ConvertNDHWCTo6HD)
                          .Add(ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, Convert6HDToNCDHW)
                          .Add(ge::FORMAT_NDC1HWC0, ge::FORMAT_NDHWC, Convert6HDToNDHWC)
                          .Add(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, ConvertHWCNToFZ)
                          .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, ConvertFZToHWCN)
                          .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, ConvertNDToFZ)
                          .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_ND, ConvertFZToND)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, ConvertNCHWToFZ)
                          .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW, ConvertFZToNCHW)
                          .Add(ge::FORMAT_DHWCN, ge::FORMAT_FRACTAL_Z_3D, ConvertDHWCNToFZ3D)
                          .Add(ge::FORMAT_NDHWC, ge::FORMAT_FRACTAL_Z_3D, ConvertNDHWCToFZ3D)
                          .Add(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ConvertNCDHWToFZ3D)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, Convert5HDToFZ)
                          .Add(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ConvertFZ3DToNCDHW)
                          .Add(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NDHWC, ConvertFZ3DToNDHWC)
                          .Add(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_DHWCN, ConvertFZ3DToDHWCN)
                          .Add(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ConvertNZTo5HD);

  return table.Find(src_format, dst_format);
}

const RealSrcDstFormat* GetRealFormat(ge::Format src_format, ge::Format dst_format) {
  static RealSrcDstFormat default_value = {RF_END, RF_END};
  static auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, RealSrcDstFormat>(default_value)
                          .Add(format_nd_array, ARRAYNDLEN, ge::FORMAT_FRACTAL_NZ, RF_HNC, RF_HCNT)
                          .Add(ge::FORMAT_FRACTAL_NZ, format_nd_array, ARRAYNDLEN, RF_HCNT, RF_HNC)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, RF_NCH, RF_NCHT)
                          .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, RF_NHC, RF_NCHT)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, RF_NCHT, RF_NCH)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC, RF_NCHT, RF_NHC)
                          .Add(ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, RF_NCDH, RF_NDCHT)
                          .Add(ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, RF_NDHC, RF_NDCHT)
                          .Add(ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, RF_NDCHT, RF_NCDH)
                          .Add(ge::FORMAT_NDC1HWC0, ge::FORMAT_NDHWC, RF_NCHT, RF_NHC)
                          .Add(ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z, RF_HCN, RF_CHNT)
                          .Add(ge::FORMAT_ND, ge::FORMAT_FRACTAL_Z, RF_HCN, RF_HCNT)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z, RF_NCH, RF_CHNT)
                          .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN, RF_CHNT, RF_HCN)
                          .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_ND, RF_HCNT, RF_HCN)
                          .Add(ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW, RF_CHNT, RF_NCH)
                          .Add(ge::FORMAT_DHWCN, ge::FORMAT_FRACTAL_Z_3D, RF_DHCN, RF_DCHNT)
                          .Add(ge::FORMAT_NDHWC, ge::FORMAT_FRACTAL_Z_3D, RF_NDHC, RF_DCHNT)
                          .Add(ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, RF_NCDH, RF_DCHNT)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, RF_NDHC, RF_DCHNT)
                          .Add(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, RF_DCHNT, RF_NCDH)
                          .Add(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NDHWC, RF_DCHNT, RF_NDHC)
                          .Add(ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_DHWCN, RF_DCHNT, RF_DHCN)
                          .Add(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, RF_DCHNT, RF_NDHC);

  return table.FindPointer(src_format, dst_format);
}

DoRealTilingFunc GetRealTilingFunc(RealFormat src_rf, RealFormat dst_rf) {
  static auto table = TableDriven2<RF_END, RF_END, DoRealTilingFunc>(nullptr)
                          .Add(RF_HNC, RF_HCNT, TillingPositiveMode1010)       // ND->NZ
                          .Add(RF_HCNT, RF_HNC, TilingNegativeTc201)           // NZ->ND
                          .Add(RF_DCHNT, RF_NDHC, TilingNegativeTc201)         // NZ->5HD
                          .Add(RF_NHC, RF_NCHT, TillingPositiveMode1010)       // NHWC->5HD
                          .Add(RF_NCHT, RF_NHC, TilingNegativeTc201)           // 5HD->NHWC, 6HD->NDHWC
                          .Add(RF_NCH, RF_NCHT, TilingPositiveSourceNtc100)    // NCHW->5HD
                          .Add(RF_NCHT, RF_NCH, TilingNegativeNtc200)          // 5HD->NCHW
                          .Add(RF_NDCHT, RF_NCDH, TilingNegativeNtc200)        // 6HD->NCDHW
                          .Add(RF_NCDH, RF_NDCHT, TilingPositiveSourceNtc100)  // NCDHW->6HD
                          .Add(RF_NDHC, RF_NDCHT, TillingPositiveMode1010)     // NDHWC->6HD
                          .Add(RF_HCN, RF_CHNT, TilingPositiveSourceNtc100)    // HWCN->FZ
                          .Add(RF_HCN, RF_HCNT, TilingPositiveSourceNtc100)    // ND->FZ
                          .Add(RF_NCH, RF_CHNT, TilingPositiveSourceNtc100)    // NCHW->FZ
                          .Add(RF_NDHC, RF_DCHNT, TillingPositiveMode1011)     // 5HD->FZ
                          .Add(RF_CHNT, RF_HCN, TilingNegativeNtc200)          // FZ->HWCN
                          .Add(RF_HCNT, RF_HCN, TilingNegativeNtc200)          // FZ->ND
                          .Add(RF_CHNT, RF_NCH, TilingNegativeNtc200)          // FZ->NCHW
                          .Add(RF_DHCN, RF_DCHNT, TilingPositiveSourceNtc100)  // DHWCN->FZ3D
                          .Add(RF_NCDH, RF_DCHNT, TilingPositiveSourceNtc100)  // NCDHW->FZ3D
                          .Add(RF_NDHC, RF_DCHNT, TillingPositiveMode1011)     // NDHWC->FZ3D
                          .Add(RF_DCHNT, RF_NCDH, TilingNegativeNtc200)        // FZ3D->NCDHW
                          .Add(RF_DCHNT, RF_NDHC, TilingNegativeTc201)         // FZ3D->NDHWC
                          .Add(RF_DCHNT, RF_DHCN, TilingNegativeNtc200);       // FZ3D->DHWCN

  return table.Find(src_rf, dst_rf);
}
}  // namespace transdata

bool CheckShape(TilingContext* context, const gert::Shape& out_shape) {
  int32_t out_dims = out_shape.GetDimNum();
  for (int32_t i = 0; i < out_dims; i++) {
    if (out_shape[i] <= 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                      "TilingForTransData CheckShape, out_shape[%d]:%ld must be > 0", i, out_shape[i]);
      return false;
    }
  }
  return true;
}

bool CheckCompileInfo(TilingParseContext* context, TransDataCompileInfo* compile_info) {
  OP_LOGD(context->GetNodeName(),
          "Parsed TransDataTiling compile info(ub_size, block_dim, group, vnc_fp32_flag): %ld, %ld, %ld, %ld",
          compile_info->ub_size, compile_info->block_dim, compile_info->group, compile_info->vnc_fp32_flag);
  if (compile_info->ub_size <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CheckCompileInfo, invalid value ub_size:%ld",
                                    compile_info->ub_size);
    return false;
  }
  if (compile_info->block_dim <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CheckCompileInfo, invalid value block_dim:%ld",
                                    compile_info->block_dim);
    return false;
  }

  return true;
}

ge::graphStatus TilingForTransData(TilingContext* context) {
  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape)
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape)
  auto compile_info = reinterpret_cast<const TransDataCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info)
  auto src_td = context->GetInputDesc(0);
  auto dst_td = context->GetOutputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td)
  OPS_CHECK_NULL_WITH_CONTEXT(context, dst_td)

  bool check_ret = CheckShape(context, out_shape->GetStorageShape());
  OP_TILING_CHECK(!check_ret,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingForTransData CheckShape failed"),
                  return ge::GRAPH_FAILED);

  auto src_format = src_td->GetStorageFormat();
  auto dst_format = dst_td->GetStorageFormat();
  auto real_formats = transdata::GetRealFormat(src_format, dst_format);
  OPS_CHECK_NULL_WITH_CONTEXT(context, real_formats);
  OP_LOGD(context->GetNodeName(), "TilingForTransData, src_format:[%s] dst_format:[%s]",
          ops::ToString(src_format).c_str(), ops::ToString(dst_format).c_str());
  OP_LOGD(context->GetNodeName(), "TilingForTransData, real_src_format:[%s] real_dst_format:[%s]",
          RealFormatToSerialString(real_formats->src).c_str(), RealFormatToSerialString(real_formats->dst).c_str());
  auto real_shape_converter = transdata::GetRealShapeConvertFunc(src_format, dst_format);
  OPS_CHECK_NULL_WITH_CONTEXT(context, real_shape_converter);

  auto real_tiling_func = transdata::GetRealTilingFunc(real_formats->src, real_formats->dst);
  OPS_CHECK_NULL_WITH_CONTEXT(context, real_tiling_func)

  auto data_type = src_td->GetDataType();
  auto c0_size = transdata::GetC0SizeWithType(data_type);
  gert::Shape real_in_shape{}, real_out_shape{};

  auto ret = real_shape_converter(in_shape->GetStorageShape(), out_shape->GetStorageShape(), c0_size, real_in_shape,
                                  real_out_shape);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "real_shape_converter failed!");
    return ret;
  }
  int64_t block_elem_cnt = transdata::BLOCK_BYTE_SIZE / GetSizeByDataType(data_type);
  context->SetBlockDim(compile_info->block_dim);
  OP_LOGD(context->GetNodeName(), "TilingForTransData, in_shape:[%s] out_shape:[%s], c0_size:%ld, block_elem_cnt:%ld",
          ops::ToString(in_shape->GetStorageShape()).c_str(), ops::ToString(out_shape->GetStorageShape()).c_str(),
          c0_size, block_elem_cnt);
  OP_LOGD(context->GetNodeName(), "TilingForTransData, real_in_shape:[%s] real_out_shape:[%s]",
          ops::ToString(real_in_shape).c_str(), ops::ToString(real_out_shape).c_str());

  return real_tiling_func(context, real_in_shape, real_out_shape, real_formats, compile_info);
}

ge::graphStatus TilingPrepareForTransdata(TilingParseContext* context) {
  auto compile_info = GetCompileInfoPtr<TransDataCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get vars in parsed_object_cinfo."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!ReadCompileItem(vars, "ub_size", compile_info->ub_size),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub_size in compile_info"),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!ReadCompileItem(vars, "block_dim", compile_info->block_dim),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get block_dim in compile_info"),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!ReadCompileItem(vars, "group", compile_info->group),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get group in compile_info"),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!ReadCompileItem(vars, "vnc_fp32_flag", compile_info->vnc_fp32_flag), compile_info->vnc_fp32_flag = 0,
                  OP_LOGI(context->GetNodeName(), "Can not find vnc_fp32_flag in json, use default value 0"));
  bool check_ret = CheckCompileInfo(context, compile_info);
  OP_TILING_CHECK(!check_ret, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CheckCompileInfo failed"),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(TransData).Tiling(TilingForTransData).TilingParse<TransDataCompileInfo>(TilingPrepareForTransdata);
}  // namespace optiling
