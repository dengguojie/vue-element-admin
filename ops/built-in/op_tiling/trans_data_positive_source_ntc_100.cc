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

namespace optiling
{

  const int32_t FRAME_LEVEL = 2;

  bool GetFullLpCnt(const int64_t &coreNum, const int64_t &srcLpCnt, int64_t &fullLpCnt) {
    int64_t tmpFullLpCnt = GetFloorDiv(srcLpCnt, coreNum) > 0 ? coreNum : 0;
    int64_t reminderLpCnt = srcLpCnt % coreNum;
    if (reminderLpCnt == 0)
    {
      tmpFullLpCnt += coreNum;
    }
    fullLpCnt = tmpFullLpCnt + reminderLpCnt;
    return true;
  }

  int64_t GetAxisIdx(std::string format, char axis) {
    size_t resValue = format.find(axis);
    if (resValue == std::string::npos) {
      resValue = 0;
      VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Axis is not in format.");
    }
    return resValue;
  }


  bool GetMcInfoPositiveNtc100(const int64_t &srcCrLpCnt, const int64_t &srcCrSize, const int64_t &srcCLpCnt,
                               const int64_t &srcCSize, const int64_t &srcClLpCnt, const int64_t &srcClSize,
                               const int64_t &coreNum, TransDataNtc100Param &params) {
    int64_t fullLpCntCr = 0;
    int64_t fullLpCntC = 0;
    int64_t fullLpCntCl = 0;

    GetFullLpCnt(coreNum, srcCrLpCnt, fullLpCntCr);
    GetFullLpCnt(coreNum, srcCLpCnt, fullLpCntC);
    GetFullLpCnt(coreNum, srcClLpCnt, fullLpCntCl);
    if (fullLpCntCl >= fullLpCntC && fullLpCntCl >= fullLpCntCr) {
      int64_t usedCoreCnt = GetCeilDiv(srcClLpCnt, GetCeilDiv(srcClLpCnt, coreNum));
      int64_t nlcClLpCnt = GetCeilDiv(srcClLpCnt, usedCoreCnt);
      int64_t lcClLpCnt = srcClLpCnt - nlcClLpCnt * (usedCoreCnt - 1);
      params.coreParams.push_back(0);                                  // mcPos
      params.coreParams.push_back(usedCoreCnt);                        // usedCoreCnt
      params.coreParams.push_back(nlcClLpCnt * params.srcClLpStepIn);  // coreStepIn
      params.coreParams.push_back(nlcClLpCnt * params.srcClLpStepOut); // coreStepOut
      params.lcParams.push_back(nlcClLpCnt);                           // nlcClLpCnt
      params.lcParams.push_back(0);                                    // nlcClLeft
      params.lcParams.push_back(srcCLpCnt);                            // nlcCLpCnt
      params.lcParams.push_back(srcCSize % params.srcCLpUnit);         // nlcCLeft
      params.lcParams.push_back(srcCrLpCnt);                           // nlcCrLpCnt
      params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);       // nlcCrLeft
      params.lcParams.push_back(lcClLpCnt);                            // lcClLpCnt
      params.lcParams.push_back(srcClSize % params.srcClLpUnit);       // lcClLeft
      params.lcParams.push_back(srcCLpCnt);                            // lcCLpCnt
      params.lcParams.push_back(srcCSize % params.srcCLpUnit);         // lcCLeft
      params.lcParams.push_back(srcCrLpCnt);                           // lcCrLpCnt
      params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);       // lcCrLeft
    } else if (fullLpCntC >= fullLpCntCr && fullLpCntC >= fullLpCntCl) {
      int64_t usedCoreCnt = GetCeilDiv(srcCLpCnt, GetCeilDiv(srcCLpCnt, coreNum));
      int64_t nlcCLpCnt = GetCeilDiv(srcCLpCnt, usedCoreCnt);
      int64_t lcCLpCnt = srcCLpCnt - nlcCLpCnt * (usedCoreCnt - 1);
      params.coreParams.push_back(1);                                // mcPos
      params.coreParams.push_back(usedCoreCnt);                      // usedCoreCnt
      params.coreParams.push_back(nlcCLpCnt * params.srcCLpStepIn);  // coreStepIn
      params.coreParams.push_back(nlcCLpCnt * params.srcCLpStepOut); // coreStepOut
      params.lcParams.push_back(srcClLpCnt);                         // nlcClLpCnt
      params.lcParams.push_back(srcClSize % params.srcClLpUnit);     // nlcClLeft
      params.lcParams.push_back(nlcCLpCnt);                          // nlcCLpCnt
      params.lcParams.push_back(0);                                  // nlcCLeft
      params.lcParams.push_back(srcCrLpCnt);                         // nlcCrLpCnt
      params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);     // nlcCrLeft
      params.lcParams.push_back(srcClLpCnt);                         // lcClLpCnt
      params.lcParams.push_back(srcClSize % params.srcClLpUnit);     // lcClLeft
      params.lcParams.push_back(lcCLpCnt);                           // lcCLpCnt
      params.lcParams.push_back(srcCSize % params.srcCLpUnit);       // lcCLeft
      params.lcParams.push_back(srcCrLpCnt);                         // lcCrLpCnt
      params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);     // lcCrLeft
    } else if (fullLpCntCr >= fullLpCntC && fullLpCntCr >= fullLpCntCl) {
      int64_t usedCoreCnt = GetCeilDiv(srcCrLpCnt, GetCeilDiv(srcCrLpCnt, coreNum));
      int64_t nlcCrLpCnt = GetCeilDiv(srcCrLpCnt, usedCoreCnt);
      int64_t lcCrLpCnt = srcCrLpCnt - nlcCrLpCnt * (usedCoreCnt - 1);
      params.coreParams.push_back(2);                                  // mcPos
      params.coreParams.push_back(usedCoreCnt);                        // usedCoreCnt
      params.coreParams.push_back(nlcCrLpCnt * params.srcCrLpStepIn);  // coreStepIn
      params.coreParams.push_back(nlcCrLpCnt * params.srcCrLpStepOut); // coreStepOut
      params.lcParams.push_back(srcClLpCnt);                           // nlcClLpCnt
      params.lcParams.push_back(srcClSize % params.srcClLpUnit);       // nlcClLeft
      params.lcParams.push_back(srcCLpCnt);                            // nlcCLpCnt
      params.lcParams.push_back(srcCSize % params.srcCLpUnit);         // nlcCLeft                       // nlcCLeft
      params.lcParams.push_back(nlcCrLpCnt);                           // nlcCrLpCnt
      params.lcParams.push_back(0);                                    // nlcCrLeft
      params.lcParams.push_back(srcClLpCnt);                           // lcClLpCnt
      params.lcParams.push_back(srcClSize % params.srcClLpUnit);       // lcClLeft
      params.lcParams.push_back(srcCLpCnt);                            // lcCLpCnt
      params.lcParams.push_back(srcCSize % params.srcCLpUnit);         // lcCLeft
      params.lcParams.push_back(lcCrLpCnt);                            // lcCrLpCnt
      params.lcParams.push_back(srcCrSize % params.srcCrLpUnit);       // lcCrLeft
    }
    return true;
  }

  bool RenewInputOutputShapeFormat(const std::vector<int64_t> &inShape, const std::vector<int64_t> &outShape,
                                   const std::string &srcFormat, const std::string &dstFormat, const int64_t &c0Len,
                                   std::vector<int64_t> &inShapeNew, std::vector<int64_t> &outShapeNew,
                                   std::string &srcFormatNew, std::string &dstFormatNew) {
    if (srcFormat == "NCDHW" && dstFormat == "NDC1HWC0") {
      if (inShape.size() != 5) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      srcFormatNew = "NCDH";
      dstFormatNew = "NDCHT";
      inShapeNew.push_back(inShape[0]);
      inShapeNew.push_back(inShape[1]);
      inShapeNew.push_back(inShape[2]);
      inShapeNew.push_back(inShape[3] * inShape[4]);
      int64_t cIdx = GetAxisIdx(srcFormat, 'C');
      int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
      outShapeNew.push_back(inShape[0]);
      outShapeNew.push_back(inShape[2]);
      outShapeNew.push_back(axisC1);
      outShapeNew.push_back(inShape[3] * inShape[4]);
      outShapeNew.push_back(c0Len);
    } else if (srcFormat == "NCHW" && dstFormat == "NC1HWC0") {
      if (inShape.size() != 4) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      srcFormatNew = "NCH";
      dstFormatNew = "NCHT";
      inShapeNew.push_back(inShape[0]);
      inShapeNew.push_back(inShape[1]);
      inShapeNew.push_back(inShape[2] * inShape[3]);
      int64_t cIdx = GetAxisIdx(srcFormat, 'C');
      int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
      outShapeNew.push_back(inShape[0]);
      outShapeNew.push_back(axisC1);
      outShapeNew.push_back(inShape[2] * inShape[3]);
      outShapeNew.push_back(c0Len);
    } else if (srcFormat == "HWCN" && (dstFormat == "FRACTAL_Z" || dstFormat == "FRACTAL_ZN")) {
      if (inShape.size() != 4) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      srcFormatNew = "HCN";
      dstFormatNew = "CHNT";
      inShapeNew.push_back(inShape[0] * inShape[1]);
      inShapeNew.push_back(inShape[2]);
      inShapeNew.push_back(inShape[3]);
      int64_t cIdx = GetAxisIdx(srcFormat, 'C');
      int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
      int64_t nIdx = GetAxisIdx(srcFormat, 'N');
      int64_t axisNo = GetCeilDiv(inShape[nIdx], NI_16);
      outShapeNew.push_back(axisC1);
      outShapeNew.push_back(inShape[0] * inShape[1]);
      outShapeNew.push_back(NI_16 * axisNo);
      outShapeNew.push_back(c0Len);
    } else if (srcFormat == "DHWCN" && dstFormat == "FRACTAL_Z_3D") {
      if (inShape.size() != 5) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      srcFormatNew = "DHCN";
      dstFormatNew = "DCHNT";
      inShapeNew.push_back(inShape[0]);
      inShapeNew.push_back(inShape[1] * inShape[2]);
      inShapeNew.push_back(inShape[3]);
      inShapeNew.push_back(inShape[4]);
      int64_t cIdx = GetAxisIdx(srcFormat, 'C');
      int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
      int64_t nIdx = GetAxisIdx(srcFormat, 'N');
      int64_t axisNo = GetCeilDiv(inShape[nIdx], NI_16);
      outShapeNew.push_back(inShape[0]);
      outShapeNew.push_back(axisC1);
      outShapeNew.push_back(inShape[1] * inShape[2]);
      outShapeNew.push_back(NI_16 * axisNo);
      outShapeNew.push_back(c0Len);
    } else if (srcFormat == "ND" && (dstFormat == "FRACTAL_Z" || dstFormat == "FRACTAL_ZN")) {
      int64_t axisH = 1;
      int64_t axisC = 1;
      int64_t axisN = 1;
      if (inShape.size() == 1) {
        axisH = 1;
        axisC = 1;
        axisN = inShape[0];
      } else if (inShape.size() == 2) {
        axisH = 1;
        axisC = inShape[0];
        axisN = inShape[1];
      } else {
        for (size_t i = 0; i < inShape.size() - 2; i++) {
          axisH *= inShape[i];
        }
        axisC = inShape[inShape.size() - 2];
        axisN = inShape[inShape.size() - 1];
      }
      srcFormatNew = "HCN";
      dstFormatNew = "HCNT";
      inShapeNew.push_back(axisH);
      inShapeNew.push_back(axisC);
      inShapeNew.push_back(axisN);
      int64_t axisC1 = GetCeilDiv(axisC, c0Len);
      int64_t axisNo = GetCeilDiv(axisN, NI_16);
      outShapeNew.push_back(axisH);
      outShapeNew.push_back(axisC1);
      outShapeNew.push_back(axisNo * NI_16);
      outShapeNew.push_back(c0Len);
    } else if (srcFormat == "NCHW" && (dstFormat == "FRACTAL_Z" || dstFormat == "FRACTAL_ZN")) {
      if (inShape.size() != 4) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      srcFormatNew = "NCH";
      dstFormatNew = "CHNT";
      inShapeNew.push_back(inShape[0]);
      inShapeNew.push_back(inShape[1]);
      inShapeNew.push_back(inShape[2] * inShape[3]);
      int64_t cIdx = GetAxisIdx(srcFormat, 'C');
      int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
      int64_t nIdx = GetAxisIdx(srcFormat, 'N');
      int64_t axisNo = GetCeilDiv(inShape[nIdx], NI_16);
      outShapeNew.push_back(axisC1);
      outShapeNew.push_back(inShape[2] * inShape[3]);
      outShapeNew.push_back(NI_16 * axisNo);
      outShapeNew.push_back(c0Len);
    } else if (srcFormat == "NCDHW" && dstFormat == "FRACTAL_Z_3D") {
      if (inShape.size() != 5) {
        VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
        return false;
      }
      srcFormatNew = "NCDH";
      dstFormatNew = "DCHNT";
      inShapeNew.push_back(inShape[0]);
      inShapeNew.push_back(inShape[1]);
      inShapeNew.push_back(inShape[2]);
      inShapeNew.push_back(inShape[3] * inShape[4]);
      int64_t cIdx = GetAxisIdx(srcFormat, 'C');
      int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
      int64_t nIdx = GetAxisIdx(srcFormat, 'N');
      int64_t axisNo = GetCeilDiv(inShape[nIdx], NI_16);
      outShapeNew.push_back(inShape[2]);
      outShapeNew.push_back(axisC1);
      outShapeNew.push_back(inShape[3] * inShape[4]);
      outShapeNew.push_back(NI_16 * axisNo);
      outShapeNew.push_back(c0Len);
    }
    return true;
  }

  bool TilingPositiveSourceNtc100(const vector<int64_t> &inShape, const vector<int64_t> &outShape,
                                  const std::string &srcFormat, const std::string &dstFormat,
                                  const int64_t &coreNum, const int64_t &blockElemCnt, const int64_t &ubSize,
                                  const int64_t &c0Len, const std::string &dType, TransDataNtc100Param &params) {
    std::string srcFormatNew;
    std::string dstFormatNew;
    std::vector<int64_t> inShapeNew;
    std::vector<int64_t> outShapeNew;
    RenewInputOutputShapeFormat(inShape, outShape, srcFormat, dstFormat, c0Len,
                                inShapeNew, outShapeNew, srcFormatNew, dstFormatNew);

    // get tiling params for using vnchwconv
    int64_t halfUbSize = c0Len == C0_16 ? ubSize / 2 : ubSize / 4;
    int64_t oneVncLineSize = halfUbSize / VNC_LINES / blockElemCnt * blockElemCnt;
    int64_t tmpUbOffset = oneVncLineSize * VNC_LINES;
    params.ubOffset = c0Len == C0_16 ? tmpUbOffset : tmpUbOffset * 2;
    params.vncLineSize = oneVncLineSize;
    params.c0Size = c0Len;

    // axis c-right tiling parameters
    params.crDims = FRAME_LEVEL;
    params.r1stSrcR2ndDstSame = 1;
    int64_t cIdx = GetAxisIdx(srcFormatNew, 'C');
    int64_t cOdx = GetAxisIdx(dstFormatNew, 'C');
    int64_t axisSrcCrSize = GetShapeSize(inShapeNew, cIdx + 1);
    int64_t tmpSrcCrLpUnit = params.vncLineSize / c0Len / blockElemCnt * blockElemCnt;
    const std::vector<std::string> dtypeList = {"float32", "int32", "uint32"};
    if (tmpSrcCrLpUnit >= axisSrcCrSize || std::find(dtypeList.begin(), dtypeList.end(), dType) != dtypeList.end()) {
      params.tilingMode = 1000;
      params.srcCrLpUnit = axisSrcCrSize > tmpSrcCrLpUnit ? tmpSrcCrLpUnit : axisSrcCrSize;
    } else {
      params.tilingMode = 1001;
      params.srcCrLpUnit = axisSrcCrSize > params.vncLineSize ? params.vncLineSize : axisSrcCrSize;
    }

    // count method: cr_idx/dst_rsize%size*dst_asize
    std::string tmpSrcCrFormat = srcFormatNew.substr(cIdx + 1);
    std::vector<int64_t> tmpSrcCrShape;
    for (uint32_t i = 0; i < tmpSrcCrFormat.length(); i++) {
      tmpSrcCrShape.push_back(inShapeNew[i + cIdx + 1]);
    }
    tmpSrcCrShape.push_back(1);
    std::reverse(tmpSrcCrFormat.begin(), tmpSrcCrFormat.end());
    for (uint32_t i = 0; i < tmpSrcCrFormat.length(); i++) {
      int64_t tmpSrcIdx = GetAxisIdx(srcFormatNew, tmpSrcCrFormat[i]);
      int64_t tmpDstIdx = GetAxisIdx(dstFormatNew, tmpSrcCrFormat[i]);
      if (i == 0) {
        params.crOutIdx0Size = inShapeNew[tmpSrcIdx];
        params.crOutIdx0DstRSize = GetShapeSize(tmpSrcCrShape, tmpSrcCrShape.size() - i - 1);
        params.crOutIdx0DstASize = GetShapeSize(outShapeNew, tmpDstIdx + 1);
      } else if (i == 1) {
        params.crOutIdx1Size = inShapeNew[tmpSrcIdx];
        params.crOutIdx1DstRSize = GetShapeSize(tmpSrcCrShape, tmpSrcCrShape.size() - i - 1);
        params.crOutIdx1DstASize = GetShapeSize(outShapeNew, tmpDstIdx + 1);
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
    if (*(srcFormatNew.rbegin()) != *(dstFormatNew.rbegin() + 1)) {
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
      int64_t tmpIdx = std::strchr(dstFormatNew.c_str(), *(srcFormatNew.rbegin())) - dstFormatNew.c_str();
      params.srcCrStepOut = GetShapeSize(outShapeNew, tmpIdx + 1);
      params.srcCrLpStepOut = params.srcCrStepOut * params.srcCrLpUnit;
    }

    // axis c tiling parameters
    int64_t axisSrcCSize = inShapeNew[cIdx];
    params.srcCLpUnit = c0Len;
    int64_t srcCLpCnt = GetCeilDiv(axisSrcCSize, params.srcCLpUnit);
    params.srcCStepIn = GetShapeSize(inShapeNew, cIdx + 1);
    params.srcCLpStepIn = params.srcCStepIn * params.srcCLpUnit;
    params.srcCLpStepOut = GetShapeSize(outShapeNew, cOdx + 1);
    params.cModC0 = axisSrcCSize % c0Len;

    // axis left parameters
    params.clDims = FRAME_LEVEL;
    int64_t axisSrcClSize = GetShapeSize(inShapeNew, 0) / GetShapeSize(inShapeNew, cIdx);
    if (params.tilingMode == 1000 ||
        (params.r1stSrcR2ndDstSame == 0 && params.tilingMode == 1001 && params.crDims != 1)) {
      params.srcClLpUnit = axisSrcClSize > NI_16 ? NI_16 : axisSrcClSize;
    } else {
      params.srcClLpUnit = 1;
    }
    int64_t srcClLpCnt = GetCeilDiv(axisSrcClSize, params.srcClLpUnit);

    // count method: left_axis_size/dst_rsize%size*asize
    std::string tmpSrcClFormat = srcFormatNew.substr(0, cIdx);
    std::vector<int64_t> tmpSrcClShape;
    for (uint32_t i = 0; i < tmpSrcClFormat.length(); i++) {
      tmpSrcClShape.push_back(inShapeNew[i]);
    }
    tmpSrcClShape.push_back(1);
    std::reverse(tmpSrcClFormat.begin(), tmpSrcClFormat.end());
    for (uint32_t i = 0; i < tmpSrcClFormat.length(); i++) {
      int64_t tmpSrcClIdx = GetAxisIdx(srcFormatNew, tmpSrcClFormat[i]);
      int64_t tmpDstClIdx = GetAxisIdx(dstFormatNew, tmpSrcClFormat[i]);
      if (i == 0) {
        params.clOutIdx0Size = inShapeNew[tmpSrcClIdx];
        params.clOutIdx0DstRSize = GetShapeSize(tmpSrcClShape, tmpSrcClShape.size() - i - 1);
        params.clOutIdx0DstASize = GetShapeSize(outShapeNew, tmpDstClIdx + 1);
      } else if (i == 1) {
        params.clOutIdx1Size = inShapeNew[tmpSrcClIdx];
        params.clOutIdx1DstRSize = GetShapeSize(tmpSrcClShape, tmpSrcClShape.size() - i - 1);
        params.clOutIdx1DstASize = GetShapeSize(outShapeNew, tmpDstClIdx + 1);
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
    params.srcClStepIn = GetShapeSize(inShapeNew, cIdx);
    params.srcClLpStepIn = params.srcClStepIn * params.srcClLpUnit;
    if (params.clDims == 2) {
      params.srcClStepOut = 0;
      params.srcClLpStepOut = 0;
    } else {
      int64_t tmpIdx = GetAxisIdx(dstFormatNew, srcFormatNew[0]);
      params.srcClStepOut = GetShapeSize(outShapeNew, tmpIdx + 1);
      params.srcClLpStepOut = params.srcClStepOut * params.srcClLpUnit;
    }

    // mulitple core parameters
    bool ret = GetMcInfoPositiveNtc100(srcCrLpCnt, axisSrcCrSize, srcCLpCnt, axisSrcCSize, srcClLpCnt, axisSrcClSize,
                                       coreNum, params);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoPositiveNtc100 Failed.");
      return ret;
    }

    return true;
  }

  void SetRunningNtc100Params(const TransDataNtc100Param &runParams, OpRunInfo &runInfo) {
    ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
    ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
    for (auto i : runParams.coreParams) {
      ByteBufferPut(runInfo.tiling_data, int64_t(i));
    }
    ByteBufferPut(runInfo.tiling_data, runParams.vncLineSize);
    ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
    ByteBufferPut(runInfo.tiling_data, runParams.c0Size);
    ByteBufferPut(runInfo.tiling_data, runParams.clDims);
    ByteBufferPut(runInfo.tiling_data, runParams.crDims);
    ByteBufferPut(runInfo.tiling_data, runParams.r1stSrcR2ndDstSame);
    ByteBufferPut(runInfo.tiling_data, runParams.srcClStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcClStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.srcClLpUnit);
    ByteBufferPut(runInfo.tiling_data, runParams.srcClLpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcClLpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCLpUnit);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCrStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCrStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCrLpUnit);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCrLpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcCrLpStepOut);
    for (auto i : runParams.lcParams) {
      ByteBufferPut(runInfo.tiling_data, int64_t(i));
    }
    ByteBufferPut(runInfo.tiling_data, runParams.clOutIdx0Size);
    ByteBufferPut(runInfo.tiling_data, runParams.clOutIdx0DstRSize);
    ByteBufferPut(runInfo.tiling_data, runParams.clOutIdx0DstASize);
    ByteBufferPut(runInfo.tiling_data, runParams.clOutIdx1Size);
    ByteBufferPut(runInfo.tiling_data, runParams.clOutIdx1DstRSize);
    ByteBufferPut(runInfo.tiling_data, runParams.clOutIdx1DstASize);
    ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx0Size);
    ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx0DstRSize);
    ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx0DstASize);
    ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx1Size);
    ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx1DstRSize);
    ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx1DstASize);
  }

} // namespace optiling
