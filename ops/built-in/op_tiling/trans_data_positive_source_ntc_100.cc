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
 * \file trans_data_positive_source_ntc_100.cc
 * \brief dynamic TransData op tiling
 */
#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "trans_data_common.h"
#include "error_log.h"

namespace optiling {

const int32_t FRAME_LEVEL = 2;

bool GetFullLpCnt(const int64_t& core_num, const int64_t& srcLpCnt, int64_t& fullLpCnt) {
  int64_t tmpFullLpCnt = GetFloorDiv(srcLpCnt, core_num) > 0 ? core_num : 0;
  int64_t reminderLpCnt = srcLpCnt % core_num;
  if (reminderLpCnt == 0) {
    tmpFullLpCnt += core_num;
  }
  fullLpCnt = tmpFullLpCnt + reminderLpCnt;
  return true;
}

int64_t GetAxisIdx(std::string data_format, char axis) {
  size_t resValue = data_format.find(axis);
  if (resValue == std::string::npos) {
    resValue = 0;
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Axis is not in data_format.");
  }
  return resValue;
}

bool GetMcInfoPositiveNtc100(const int64_t& srcCrLpCnt, const int64_t& srcCrSize, const int64_t& srcCLpCnt,
                             const int64_t& srcCSize, const int64_t& srcClLpCnt, const int64_t& srcClSize,
                             const int64_t& core_num, TransDataNtc100Param& params) {
  int64_t fullLpCntCr = 0;
  int64_t fullLpCntC = 0;
  int64_t fullLpCntCl = 0;

  GetFullLpCnt(core_num, srcCrLpCnt, fullLpCntCr);
  GetFullLpCnt(core_num, srcCLpCnt, fullLpCntC);
  GetFullLpCnt(core_num, srcClLpCnt, fullLpCntCl);
  if (fullLpCntCl >= fullLpCntC && fullLpCntCl >= fullLpCntCr) {
    int64_t usedCoreCnt = GetCeilDiv(srcClLpCnt, GetCeilDiv(srcClLpCnt, core_num));
    int64_t nlcClLpCnt = GetCeilDiv(srcClLpCnt, usedCoreCnt);
    int64_t lcClLpCnt = srcClLpCnt - nlcClLpCnt * (usedCoreCnt - 1);
    params.coreParams.push_back(0);                                   // mcPos
    params.coreParams.push_back(usedCoreCnt);                         // usedCoreCnt
    params.coreParams.push_back(nlcClLpCnt * params.srcClLpStepIn);   // coreStepIn
    params.coreParams.push_back(nlcClLpCnt * params.srcClLpStepOut);  // coreStepOut
    params.lcParams.push_back(nlcClLpCnt);                            // nlcClLpCnt
    params.lcParams.push_back(0);                                     // nlcClLeft
    params.lcParams.push_back(srcCLpCnt);                             // nlcCLpCnt
    params.lcParams.push_back(srcCSize % params.srcCLpUnit);          // nlcCLeft
    params.lcParams.push_back(srcCrLpCnt);                            // nlcCrLpCnt
    params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);        // nlcCrLeft
    params.lcParams.push_back(lcClLpCnt);                             // lcClLpCnt
    params.lcParams.push_back(srcClSize % params.srcClLpUnit);        // lcClLeft
    params.lcParams.push_back(srcCLpCnt);                             // lcCLpCnt
    params.lcParams.push_back(srcCSize % params.srcCLpUnit);          // lcCLeft
    params.lcParams.push_back(srcCrLpCnt);                            // lcCrLpCnt
    params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);        // lcCrLeft
  } else if (fullLpCntC >= fullLpCntCr && fullLpCntC >= fullLpCntCl) {
    int64_t usedCoreCnt = GetCeilDiv(srcCLpCnt, GetCeilDiv(srcCLpCnt, core_num));
    int64_t nlcCLpCnt = GetCeilDiv(srcCLpCnt, usedCoreCnt);
    int64_t lcCLpCnt = srcCLpCnt - nlcCLpCnt * (usedCoreCnt - 1);
    params.coreParams.push_back(1);                                 // mcPos
    params.coreParams.push_back(usedCoreCnt);                       // usedCoreCnt
    params.coreParams.push_back(nlcCLpCnt * params.srcCLpStepIn);   // coreStepIn
    params.coreParams.push_back(nlcCLpCnt * params.srcCLpStepOut);  // coreStepOut
    params.lcParams.push_back(srcClLpCnt);                          // nlcClLpCnt
    params.lcParams.push_back(srcClSize % params.srcClLpUnit);      // nlcClLeft
    params.lcParams.push_back(nlcCLpCnt);                           // nlcCLpCnt
    params.lcParams.push_back(0);                                   // nlcCLeft
    params.lcParams.push_back(srcCrLpCnt);                          // nlcCrLpCnt
    params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);      // nlcCrLeft
    params.lcParams.push_back(srcClLpCnt);                          // lcClLpCnt
    params.lcParams.push_back(srcClSize % params.srcClLpUnit);      // lcClLeft
    params.lcParams.push_back(lcCLpCnt);                            // lcCLpCnt
    params.lcParams.push_back(srcCSize % params.srcCLpUnit);        // lcCLeft
    params.lcParams.push_back(srcCrLpCnt);                          // lcCrLpCnt
    params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);      // lcCrLeft
  } else if (fullLpCntCr >= fullLpCntC && fullLpCntCr >= fullLpCntCl) {
    int64_t usedCoreCnt = GetCeilDiv(srcCrLpCnt, GetCeilDiv(srcCrLpCnt, core_num));
    int64_t nlcCrLpCnt = GetCeilDiv(srcCrLpCnt, usedCoreCnt);
    int64_t lcCrLpCnt = srcCrLpCnt - nlcCrLpCnt * (usedCoreCnt - 1);
    params.coreParams.push_back(2);                                   // mcPos
    params.coreParams.push_back(usedCoreCnt);                         // usedCoreCnt
    params.coreParams.push_back(nlcCrLpCnt * params.srcCrLpStepIn);   // coreStepIn
    params.coreParams.push_back(nlcCrLpCnt * params.srcCrLpStepOut);  // coreStepOut
    params.lcParams.push_back(srcClLpCnt);                            // nlcClLpCnt
    params.lcParams.push_back(srcClSize % params.srcClLpUnit);        // nlcClLeft
    params.lcParams.push_back(srcCLpCnt);                             // nlcCLpCnt
    params.lcParams.push_back(srcCSize % params.srcCLpUnit);          // nlcCLeft                       // nlcCLeft
    params.lcParams.push_back(nlcCrLpCnt);                            // nlcCrLpCnt
    params.lcParams.push_back(0);                                     // nlcCrLeft
    params.lcParams.push_back(srcClLpCnt);                            // lcClLpCnt
    params.lcParams.push_back(srcClSize % params.srcClLpUnit);        // lcClLeft
    params.lcParams.push_back(srcCLpCnt);                             // lcCLpCnt
    params.lcParams.push_back(srcCSize % params.srcCLpUnit);          // lcCLeft
    params.lcParams.push_back(lcCrLpCnt);                             // lcCrLpCnt
    params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);        // lcCrLeft
  }
  return true;
}

bool RenewInputOutputShapeFormat(const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                                 const ge::Format& src_format, const ge::Format& dst_format, const int64_t& c0Len,
                                 std::vector<int64_t>& in_shape_new, std::vector<int64_t>& out_shape_new,
                                 std::string& src_format_new, std::string& dst_format_new) {
  if (src_format == FORMAT_NCDHW && dst_format == FORMAT_NDC1HWC0) {
    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    src_format_new = "NCDH";
    dst_format_new = "NDCHT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[2]);
    in_shape_new.push_back(in_shape[3] * in_shape[4]);
    int64_t cIdx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axisC1 = GetCeilDiv(in_shape[cIdx], c0Len);
    out_shape_new.push_back(in_shape[0]);
    out_shape_new.push_back(in_shape[2]);
    out_shape_new.push_back(axisC1);
    out_shape_new.push_back(in_shape[3] * in_shape[4]);
    out_shape_new.push_back(c0Len);
  } else if (src_format == FORMAT_NCHW && dst_format == FORMAT_NC1HWC0) {
    if (in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    src_format_new = "NCH";
    dst_format_new = "NCHT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[2] * in_shape[3]);
    int64_t cIdx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axisC1 = GetCeilDiv(in_shape[cIdx], c0Len);
    out_shape_new.push_back(in_shape[0]);
    out_shape_new.push_back(axisC1);
    out_shape_new.push_back(in_shape[2] * in_shape[3]);
    out_shape_new.push_back(c0Len);
  } else if (src_format == FORMAT_HWCN && dst_format == FORMAT_FRACTAL_Z) {
    if (in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    src_format_new = "HCN";
    dst_format_new = "CHNT";
    in_shape_new.push_back(in_shape[0] * in_shape[1]);
    in_shape_new.push_back(in_shape[2]);
    in_shape_new.push_back(in_shape[3]);
    int64_t cIdx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axisC1 = GetCeilDiv(in_shape[cIdx], c0Len);
    int64_t nIdx = GetIdxFromFormat(N_IDX_MAP, src_format);
    int64_t axisNo = GetCeilDiv(in_shape[nIdx], NI_16);
    out_shape_new.push_back(axisC1);
    out_shape_new.push_back(in_shape[0] * in_shape[1]);
    out_shape_new.push_back(NI_16 * axisNo);
    out_shape_new.push_back(c0Len);
  } else if (src_format == FORMAT_DHWCN && dst_format == FORMAT_FRACTAL_Z_3D) {
    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    src_format_new = "DHCN";
    dst_format_new = "DCHNT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1] * in_shape[2]);
    in_shape_new.push_back(in_shape[3]);
    in_shape_new.push_back(in_shape[4]);
    int64_t cIdx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axisC1 = GetCeilDiv(in_shape[cIdx], c0Len);
    int64_t nIdx = GetIdxFromFormat(N_IDX_MAP, src_format);
    int64_t axisNo = GetCeilDiv(in_shape[nIdx], NI_16);
    out_shape_new.push_back(in_shape[0]);
    out_shape_new.push_back(axisC1);
    out_shape_new.push_back(in_shape[1] * in_shape[2]);
    out_shape_new.push_back(NI_16 * axisNo);
    out_shape_new.push_back(c0Len);
  } else if (src_format == FORMAT_ND && dst_format == FORMAT_FRACTAL_Z) {
    int64_t axisH = 1;
    int64_t axisC = 1;
    int64_t axisN = 1;
    if (in_shape.size() == 1) {
      axisH = 1;
      axisC = 1;
      axisN = in_shape[0];
    } else if (in_shape.size() == 2) {
      axisH = 1;
      axisC = in_shape[0];
      axisN = in_shape[1];
    } else {
      for (size_t i = 0; i < in_shape.size() - 2; i++) {
        axisH *= in_shape[i];
      }
      axisC = in_shape[in_shape.size() - 2];
      axisN = in_shape[in_shape.size() - 1];
    }
    src_format_new = "HCN";
    dst_format_new = "HCNT";
    in_shape_new.push_back(axisH);
    in_shape_new.push_back(axisC);
    in_shape_new.push_back(axisN);
    int64_t axisC1 = GetCeilDiv(axisC, c0Len);
    int64_t axisNo = GetCeilDiv(axisN, NI_16);
    out_shape_new.push_back(axisH);
    out_shape_new.push_back(axisC1);
    out_shape_new.push_back(axisNo * NI_16);
    out_shape_new.push_back(c0Len);
  } else if (src_format == FORMAT_NCHW && dst_format == FORMAT_FRACTAL_Z) {
    if (in_shape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    src_format_new = "NCH";
    dst_format_new = "CHNT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[2] * in_shape[3]);
    int64_t cIdx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axisC1 = GetCeilDiv(in_shape[cIdx], c0Len);
    int64_t nIdx = GetIdxFromFormat(N_IDX_MAP, src_format);
    int64_t axisNo = GetCeilDiv(in_shape[nIdx], NI_16);
    out_shape_new.push_back(axisC1);
    out_shape_new.push_back(in_shape[2] * in_shape[3]);
    out_shape_new.push_back(NI_16 * axisNo);
    out_shape_new.push_back(c0Len);
  } else if (src_format == FORMAT_NCDHW && dst_format == FORMAT_FRACTAL_Z_3D) {
    if (in_shape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    src_format_new = "NCDH";
    dst_format_new = "DCHNT";
    in_shape_new.push_back(in_shape[0]);
    in_shape_new.push_back(in_shape[1]);
    in_shape_new.push_back(in_shape[2]);
    in_shape_new.push_back(in_shape[3] * in_shape[4]);
    int64_t cIdx = GetIdxFromFormat(C_IDX_MAP, src_format);
    int64_t axisC1 = GetCeilDiv(in_shape[cIdx], c0Len);
    int64_t nIdx = GetIdxFromFormat(N_IDX_MAP, src_format);
    int64_t axisNo = GetCeilDiv(in_shape[nIdx], NI_16);
    out_shape_new.push_back(in_shape[2]);
    out_shape_new.push_back(axisC1);
    out_shape_new.push_back(in_shape[3] * in_shape[4]);
    out_shape_new.push_back(NI_16 * axisNo);
    out_shape_new.push_back(c0Len);
  }
  return true;
}

bool TilingPositiveSourceNtc100(const vector<int64_t>& in_shape, const vector<int64_t>& out_shape,
                                const ge::Format& src_format, const ge::Format& dst_format, const int64_t& core_num,
                                const int64_t& block_elem_cnt, const int64_t& ub_size, const int64_t& c0Len,
                                const DataType& dType, TransDataNtc100Param& params) {
  std::string src_format_new;
  std::string dst_format_new;
  std::vector<int64_t> in_shape_new;
  std::vector<int64_t> out_shape_new;
  RenewInputOutputShapeFormat(in_shape, out_shape, src_format, dst_format, c0Len, in_shape_new, out_shape_new,
                              src_format_new, dst_format_new);

  // get tiling params for using vnchwconv
  int64_t half_ub_size = c0Len == C0_16 ? ub_size / 2 : ub_size / 4;
  int64_t oneVncLineSize = half_ub_size / VNC_LINES / block_elem_cnt * block_elem_cnt;
  int64_t tmpUbOffset = oneVncLineSize * VNC_LINES;
  params.ubOffset = c0Len == C0_16 ? tmpUbOffset : tmpUbOffset * 2;
  params.vncLineSize = oneVncLineSize;
  params.c0Size = c0Len;

  // axis c-right tiling parameters
  params.crDims = FRAME_LEVEL;
  params.r1stSrcR2ndDstSame = 1;
  int64_t cIdx = GetAxisIdx(src_format_new, 'C');
  int64_t cOdx = GetAxisIdx(dst_format_new, 'C');
  int64_t axisSrcCrSize = GetShapeSize(in_shape_new, cIdx + 1);
  int64_t tmpSrcCrLpUnit = params.vncLineSize / c0Len / block_elem_cnt * block_elem_cnt;
  const std::vector<DataType> dtypeList = {ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32};
  if (tmpSrcCrLpUnit >= axisSrcCrSize || std::find(dtypeList.begin(), dtypeList.end(), dType) != dtypeList.end()) {
    params.tilingMode = 1000;
    params.srcCrLpUnit = axisSrcCrSize > tmpSrcCrLpUnit ? tmpSrcCrLpUnit : axisSrcCrSize;
  } else {
    params.tilingMode = 1001;
    params.srcCrLpUnit = axisSrcCrSize > params.vncLineSize ? params.vncLineSize : axisSrcCrSize;
  }

  // count method: cr_idx/dst_rsize%size*dst_asize
  std::string tmpSrcCrFormat = src_format_new.substr(cIdx + 1);
  std::vector<int64_t> tmpSrcCrShape;
  for (uint32_t i = 0; i < tmpSrcCrFormat.length(); i++) {
    tmpSrcCrShape.push_back(in_shape_new[i + cIdx + 1]);
  }
  tmpSrcCrShape.push_back(1);
  std::reverse(tmpSrcCrFormat.begin(), tmpSrcCrFormat.end());
  for (uint32_t i = 0; i < tmpSrcCrFormat.length(); i++) {
    int64_t tmpSrcIdx = GetAxisIdx(src_format_new, tmpSrcCrFormat[i]);
    int64_t tmpDstIdx = GetAxisIdx(dst_format_new, tmpSrcCrFormat[i]);
    if (i == 0) {
      params.crOutIdx0Size = in_shape_new[tmpSrcIdx];
      params.crOutIdx0DstRSize = GetShapeSize(tmpSrcCrShape, tmpSrcCrShape.size() - i - 1);
      params.crOutIdx0DstASize = GetShapeSize(out_shape_new, tmpDstIdx + 1);
    } else if (i == 1) {
      params.crOutIdx1Size = in_shape_new[tmpSrcIdx];
      params.crOutIdx1DstRSize = GetShapeSize(tmpSrcCrShape, tmpSrcCrShape.size() - i - 1);
      params.crOutIdx1DstASize = GetShapeSize(out_shape_new, tmpDstIdx + 1);
    }
  }

  // suppose there are 2 axises
  int64_t padAxisCnt = FRAME_LEVEL - tmpSrcCrFormat.length();
  if (padAxisCnt) {
    params.crDims = 1;
    params.crOutIdx1Size = 1;
    params.crOutIdx1DstRSize = 1;
    params.crOutIdx1DstASize = 0;
  }
  if (*(src_format_new.rbegin()) != *(dst_format_new.rbegin() + 1)) {
    params.r1stSrcR2ndDstSame = 0;
    if (params.tilingMode == 1001 && params.crDims != 1) {
      int64_t tmpLpUnit = oneVncLineSize / NI_16;
      params.srcCrLpUnit = axisSrcCrSize > tmpLpUnit ? tmpLpUnit : axisSrcCrSize;
    }
  }
  int64_t srcCrLpCnt = GetCeilDiv(axisSrcCrSize, params.srcCrLpUnit);
  params.srcCrStepIn = 1;
  params.srcCrLpStepIn = params.srcCrStepIn * params.srcCrLpUnit;
  if (params.crDims == 2) {
    params.srcCrStepOut = 0;
    params.srcCrLpStepOut = 0;
  } else {
    int64_t tmpIdx = std::strchr(dst_format_new.c_str(), *(src_format_new.rbegin())) - dst_format_new.c_str();
    params.srcCrStepOut = GetShapeSize(out_shape_new, tmpIdx + 1);
    params.srcCrLpStepOut = params.srcCrStepOut * params.srcCrLpUnit;
  }

  // axis c tiling parameters
  int64_t axisSrcCSize = in_shape_new[cIdx];
  params.srcCLpUnit = c0Len;
  int64_t srcCLpCnt = GetCeilDiv(axisSrcCSize, params.srcCLpUnit);
  params.srcCStepIn = GetShapeSize(in_shape_new, cIdx + 1);
  params.srcCLpStepIn = params.srcCStepIn * params.srcCLpUnit;
  params.srcCLpStepOut = GetShapeSize(out_shape_new, cOdx + 1);
  params.cModC0 = axisSrcCSize % c0Len;

  // axis left parameters
  params.clDims = FRAME_LEVEL;
  int64_t axisSrcClSize = GetShapeSize(in_shape_new, 0) / GetShapeSize(in_shape_new, cIdx);
  if (params.tilingMode == 1000 ||
      (params.r1stSrcR2ndDstSame == 0 && params.tilingMode == 1001 && params.crDims != 1)) {
    params.srcClLpUnit = axisSrcClSize > NI_16 ? NI_16 : axisSrcClSize;
  } else {
    params.srcClLpUnit = 1;
  }
  int64_t srcClLpCnt = GetCeilDiv(axisSrcClSize, params.srcClLpUnit);

  // count method: left_axis_size/dst_rsize%size*asize
  std::string tmpSrcClFormat = src_format_new.substr(0, cIdx);
  std::vector<int64_t> tmpSrcClShape;
  for (uint32_t i = 0; i < tmpSrcClFormat.length(); i++) {
    tmpSrcClShape.push_back(in_shape_new[i]);
  }
  tmpSrcClShape.push_back(1);
  std::reverse(tmpSrcClFormat.begin(), tmpSrcClFormat.end());
  for (uint32_t i = 0; i < tmpSrcClFormat.length(); i++) {
    int64_t tmpSrcClIdx = GetAxisIdx(src_format_new, tmpSrcClFormat[i]);
    int64_t tmpDstClIdx = GetAxisIdx(dst_format_new, tmpSrcClFormat[i]);
    if (i == 0) {
      params.clOutIdx0Size = in_shape_new[tmpSrcClIdx];
      params.clOutIdx0DstRSize = GetShapeSize(tmpSrcClShape, tmpSrcClShape.size() - i - 1);
      params.clOutIdx0DstASize = GetShapeSize(out_shape_new, tmpDstClIdx + 1);
    } else if (i == 1) {
      params.clOutIdx1Size = in_shape_new[tmpSrcClIdx];
      params.clOutIdx1DstRSize = GetShapeSize(tmpSrcClShape, tmpSrcClShape.size() - i - 1);
      params.clOutIdx1DstASize = GetShapeSize(out_shape_new, tmpDstClIdx + 1);
    }
  }

  // suppose there are 2 axises
  padAxisCnt = FRAME_LEVEL - tmpSrcClFormat.length();
  if (padAxisCnt) {
    params.clDims = 1;
    params.clOutIdx1Size = 1;
    params.clOutIdx1DstRSize = 1;
    params.clOutIdx1DstASize = 0;
  }
  params.srcClStepIn = GetShapeSize(in_shape_new, cIdx);
  params.srcClLpStepIn = params.srcClStepIn * params.srcClLpUnit;
  if (params.clDims == 2) {
    params.srcClStepOut = 0;
    params.srcClLpStepOut = 0;
  } else {
    int64_t tmpIdx = GetAxisIdx(dst_format_new, src_format_new[0]);
    params.srcClStepOut = GetShapeSize(out_shape_new, tmpIdx + 1);
    params.srcClLpStepOut = params.srcClStepOut * params.srcClLpUnit;
  }

  // mulitple core parameters
  bool ret = GetMcInfoPositiveNtc100(srcCrLpCnt, axisSrcCrSize, srcCLpCnt, axisSrcCSize, srcClLpCnt, axisSrcClSize,
                                     core_num, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoPositiveNtc100 Failed.");
    return ret;
  }

  return true;
}

void SetRunningNtc100Params(const TransDataNtc100Param& runParams, utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(runParams.tilingMode);
  runInfo.AddTilingData(runParams.ubOffset);
  for (auto i : runParams.coreParams) {
    runInfo.AddTilingData(int64_t(i));
  }
  runInfo.AddTilingData(runParams.vncLineSize);
  runInfo.AddTilingData(runParams.cModC0);
  runInfo.AddTilingData(runParams.c0Size);
  runInfo.AddTilingData(runParams.clDims);
  runInfo.AddTilingData(runParams.crDims);
  runInfo.AddTilingData(runParams.r1stSrcR2ndDstSame);
  runInfo.AddTilingData(runParams.srcClStepIn);
  runInfo.AddTilingData(runParams.srcClStepOut);
  runInfo.AddTilingData(runParams.srcClLpUnit);
  runInfo.AddTilingData(runParams.srcClLpStepIn);
  runInfo.AddTilingData(runParams.srcClLpStepOut);
  runInfo.AddTilingData(runParams.srcCStepIn);
  runInfo.AddTilingData(runParams.srcCLpUnit);
  runInfo.AddTilingData(runParams.srcCLpStepIn);
  runInfo.AddTilingData(runParams.srcCLpStepOut);
  runInfo.AddTilingData(runParams.srcCrStepIn);
  runInfo.AddTilingData(runParams.srcCrStepOut);
  runInfo.AddTilingData(runParams.srcCrLpUnit);
  runInfo.AddTilingData(runParams.srcCrLpStepIn);
  runInfo.AddTilingData(runParams.srcCrLpStepOut);
  for (auto i : runParams.lcParams) {
    runInfo.AddTilingData(int64_t(i));
  }
  runInfo.AddTilingData(runParams.clOutIdx0Size);
  runInfo.AddTilingData(runParams.clOutIdx0DstRSize);
  runInfo.AddTilingData(runParams.clOutIdx0DstASize);
  runInfo.AddTilingData(runParams.clOutIdx1Size);
  runInfo.AddTilingData(runParams.clOutIdx1DstRSize);
  runInfo.AddTilingData(runParams.clOutIdx1DstASize);
  runInfo.AddTilingData(runParams.crOutIdx0Size);
  runInfo.AddTilingData(runParams.crOutIdx0DstRSize);
  runInfo.AddTilingData(runParams.crOutIdx0DstASize);
  runInfo.AddTilingData(runParams.crOutIdx1Size);
  runInfo.AddTilingData(runParams.crOutIdx1DstRSize);
  runInfo.AddTilingData(runParams.crOutIdx1DstASize);
}

}  // namespace optiling
