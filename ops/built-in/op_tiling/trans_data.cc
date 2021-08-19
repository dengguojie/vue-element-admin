/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "trans_data_common.h"
#include "transpose.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"

namespace optiling {

int64_t GetC0Len(std::string& opType) {
  if (opType == "int8" || opType == "uint8" || opType == "bool") {
    return C0_32;
  }
  return C0_16;
}

int64_t GetDTypeLen(std::string& opType) {
  int64_t typeLen = 1;
  if (opType == "int8" || opType == "uint8") {
    typeLen = 1;
  } else if (opType == "float16" || opType == "int16" || opType == "uint16" || opType == "bool") {
    typeLen = 2;
  } else if (opType == "float32" || opType == "int32" || opType == "uint32" || opType == "float") {
    typeLen = 4;
  } else if (opType == "int64" || opType == "uint64") {
    typeLen = 8;
  }

  return typeLen;
}

bool CheckTensorShape(const std::string& opType, int64_t ubSize, int64_t blockDim, std::vector<int64_t> outShape) {
  int32_t outDims = outShape.size();

  if (ubSize < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [TransDataTiling] : CheckTensorShape, ubSize is invalid.");
    return false;
  }

  if (blockDim < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [TransDataTiling] : CheckTensorShape, blockDim is invalid.");
    return false;
  }

  if (outDims == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [TransDataTiling] : CheckTensorShape, outShape is invalid.");
    return false;
  }

  for (int32_t i = 0; i < outDims; i++) {
    if (outShape[i] <= 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [TransDataTiling] : CheckTensorShape, outShape.shape[i] must be > 0");
      return false;
    }
  }

  return true;
}

bool GetCompileParams(const nlohmann::json& opCompileInfoJson, std::string& srcFormat, std::string& dstFormat,
                      std::string& dType, int64_t& ubSize, int64_t& blockDim, int64_t& inputSize, int64_t& hiddenSize,
                      int64_t& group, const std::string& opType) {
  using namespace nlohmann;

  auto allVars = opCompileInfoJson["vars"];
  if (allVars.count("srcFormat") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get srcFormat error");
    return false;
  }
  srcFormat = allVars["srcFormat"].get<std::string>();

  if (allVars.count("dstFormat") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get dstFormat error");
    return false;
  }
  dstFormat = allVars["dstFormat"].get<std::string>();

  if (allVars.count("dType") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get dType error");
    return false;
  }
  dType = allVars["dType"].get<std::string>();

  if (allVars.count("ubSize") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get ubSize error");
    return false;
  }
  ubSize = allVars["ubSize"].get<std::int64_t>();

  if (allVars.count("blockDim") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get blockDim error");
    return false;
  }
  blockDim = allVars["blockDim"].get<std::int64_t>();

  if (allVars.count("inputSize") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get inputSize error");
    return false;
  }
  inputSize = allVars["inputSize"].get<std::int64_t>();

  if (allVars.count("hiddenSize") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get hiddenSize error");
    return false;
  }
  hiddenSize = allVars["hiddenSize"].get<std::int64_t>();

  if (allVars.count("group") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetCompileParams, get group error");
    return false;
  }
  group = allVars["group"].get<std::int64_t>();

  if (blockDim == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "Core count cannot be zero!");
    return false;
  }

  OP_LOGD(opType.c_str(), "GetCompileParams, srcFormat[%s], dstFormat[%s], \
          dType[%s], ubSize[%d], blockDim[%d], inputSize[%d], hiddenSize[%d], group[%d].",
          srcFormat.c_str(), dstFormat.c_str(), dType.c_str(), ubSize, blockDim, inputSize, hiddenSize, group);

  return true;
}

bool GetRenew2Shape(std::vector<int64_t> inShape, std::vector<int64_t> outShape, std::string& srcFormat,
                    std::string& dstFormat, std::vector<int64_t>& combAxis, int64_t c0Len, int64_t group,
                    std::vector<int64_t>& inShapeNew, std::vector<int64_t>& outShapeNew, std::string& realSrcFormat,
                    std::string& realDstFormat) {
  int32_t combAxisCnt = combAxis.size();
  if (combAxisCnt > 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, combAxisCnt > 0");
    return false;
  }

  if ((srcFormat == "NCHW" || srcFormat == "NHWC") && (dstFormat == "NC1HWC0")) {
    int64_t hwIdx = std::strchr(srcFormat.c_str(), 'H') - srcFormat.c_str();
    int64_t cIdx = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
    realDstFormat = "NCHT";
    if (srcFormat == "NCHW") {
      realSrcFormat = "NCH";
      for (size_t i = 0; i < inShape.size() - hwIdx; i++) {
        inShapeNew.push_back(inShape[i]);
      }
      int64_t lastSize = GetShapeSize(inShape, hwIdx);
      inShapeNew.push_back(lastSize);
    } else {
      if (inShape.size() < 1) {
        VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, inShape size < 1");
        return false;
      }
      realSrcFormat = "NHC";
      for (int64_t i = 0; i < hwIdx; i++) {
        inShapeNew.push_back(inShape[i]);
      }
      int64_t n = inShape.size() - 1;
      int64_t shapeSize = 1;
      for (int64_t i = hwIdx; i < n; i++) {
        shapeSize *= inShape[i];
      }
      inShapeNew.push_back(shapeSize);
      inShapeNew.push_back(inShape[inShape.size() - 1]);
    }
    int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
    int64_t axisN = inShape[0];
    int64_t axisH = inShapeNew[hwIdx];
    int64_t axisC0 = c0Len;
    outShapeNew.push_back(axisN);
    outShapeNew.push_back(axisC1);
    outShapeNew.push_back(axisH);
    outShapeNew.push_back(axisC0);
  }

  if ((srcFormat == "ND" || srcFormat == "NHWC" || srcFormat == "NCHW") && (dstFormat == "FRACTAL_NZ")) {
    realSrcFormat = "HNC";
    realDstFormat = "HCNT";
    int64_t axisN;
    int64_t axisH;
    int64_t axisC;
    if (inShape.size() == 1) {
      axisH = 1;
      axisN = 1;
      axisC = inShape[0];
    } else if (inShape.size() == 2) {
      axisH = 1;
      axisN = inShape[0];
      axisC = inShape[1];
    } else {
      int64_t shapeSize = 1;
      for (size_t i = 0; i < inShape.size() - 2; i++) {
        shapeSize *= inShape[i];
      }
      axisH = shapeSize;
      axisN = inShape[inShape.size() - 2];
      axisC = inShape[inShape.size() - 1];
    }
    inShapeNew.push_back(axisH);
    inShapeNew.push_back(axisN);
    inShapeNew.push_back(axisC);
    int64_t axisC0 = c0Len;
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    int64_t axisNi = NI_16;
    int64_t axisNo = GetCeilDiv(axisN, axisNi);
    outShapeNew.push_back(axisH);
    outShapeNew.push_back(axisC1);
    outShapeNew.push_back(axisNo * axisNi);
    outShapeNew.push_back(axisC0);
  }

  if (srcFormat == "NC1HWC0" && dstFormat == "NCHW") {
    if (inShape.size() < 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, inShape size < 5");
      return false;
    }
    realSrcFormat = "NCHT";
    realDstFormat = "NCH";
    inShapeNew.push_back(inShape[0]);
    inShapeNew.push_back(inShape[1]);
    inShapeNew.push_back(inShape[2] * inShape[3]);
    inShapeNew.push_back(inShape[4]);

    int64_t axisC = outShape[1];
    outShapeNew.push_back(inShape[0]);
    outShapeNew.push_back(axisC);
    outShapeNew.push_back(inShape[2] * inShape[3]);
  }

  if (srcFormat == "NCDHW" && dstFormat == "NDC1HWC0") {
    realSrcFormat = "NCDH";
    realDstFormat = "NDCHT";

    if (inShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    inShapeNew.push_back(inShape[0]);
    inShapeNew.push_back(inShape[1]);
    inShapeNew.push_back(inShape[2]);
    inShapeNew.push_back(inShape[3] * inShape[4]);
    int64_t cIdx = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
    int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
    int64_t axisC0 = c0Len;
    outShapeNew.push_back(inShape[0]);
    outShapeNew.push_back(inShape[2]);
    outShapeNew.push_back(axisC1);
    outShapeNew.push_back(inShape[3] * inShape[4]);
    outShapeNew.push_back(axisC0);
  }

  if ((srcFormat == "HWCN") && (dstFormat == "FRACTAL_Z" || dstFormat == "FRACTAL_ZN")) {
    realSrcFormat = "HCN";
    realDstFormat = "CHNT";

    if (inShape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    if (outShape.size() < 2) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The output shape dimension size is not correct!");
      return false;
    }
    inShapeNew.push_back(inShape[0] * inShape[1]);
    inShapeNew.push_back(inShape[2]);
    inShapeNew.push_back(inShape[3]);

    int64_t cIdx = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
    int64_t axisC0 = outShape[outShape.size() - 1];
    int64_t axisNi = outShape[outShape.size() - 2];
    int64_t axisC1 = GetCeilDiv(inShape[cIdx], axisC0);
    int64_t axisNo = GetCeilDiv(inShape[3], axisNi);
    outShapeNew.push_back(axisC1);
    outShapeNew.push_back(inShape[0] * inShape[1]);
    outShapeNew.push_back(axisNi * axisNo);
    outShapeNew.push_back(axisC0);
  }

  if (srcFormat == "DHWCN" && dstFormat == "FRACTAL_Z_3D") {
    if (inShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    realSrcFormat = "DHCN";
    realDstFormat = "DCHNT";
    inShapeNew.push_back(inShape[0]);
    inShapeNew.push_back(inShape[1] * inShape[2]);
    inShapeNew.push_back(inShape[3]);
    inShapeNew.push_back(inShape[4]);
    int64_t cIdx = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
    int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
    int64_t axisC0 = c0Len;
    int64_t axisNi = NI_16;
    int64_t axisNo = GetCeilDiv(inShape[4], axisNi);
    outShapeNew.push_back(inShape[0]);
    outShapeNew.push_back(axisC1);
    outShapeNew.push_back(inShape[1] * inShape[2]);
    outShapeNew.push_back(axisNo * axisNi);
    outShapeNew.push_back(axisC0);
  }

  if (srcFormat == "NDHWC" && dstFormat == "FRACTAL_Z_3D") {
    if (inShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    realSrcFormat = "NDHC";
    realDstFormat = "DCHNT";
    inShapeNew.push_back(inShape[0]);
    inShapeNew.push_back(inShape[1]);
    inShapeNew.push_back(inShape[3] * inShape[2]);
    inShapeNew.push_back(inShape[4]);
    int64_t cIdx = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
    int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
    int64_t axisC0 = c0Len;
    int64_t axisNi = NI_16;
    int64_t axisNo = GetCeilDiv(inShape[0], axisNi);
    outShapeNew.push_back(inShape[1]);
    outShapeNew.push_back(axisC1);
    outShapeNew.push_back(inShape[3] * inShape[2]);
    outShapeNew.push_back(axisNo * axisNi);
    outShapeNew.push_back(axisC0);
  }

  if (srcFormat == "NC1HWC0" && dstFormat == "FRACTAL_Z") {
    if (inShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    realSrcFormat = "NDHC";
    realDstFormat = "DCHNT";
    int64_t axisN = inShape[0];
    int64_t axisC1 = inShape[1];
    int64_t axisH = inShape[2];
    int64_t axisW = inShape[3];
    int64_t axisC0 = inShape[4];
    int64_t axisD = 1;
    int64_t axisC = 1;
    int64_t axisNi = NI_16;
    int64_t axisNo = GetCeilDiv(axisN, axisNi);
    inShapeNew = {axisN, axisD, axisC1 * axisH * axisW, axisC0};
    outShapeNew = {axisD, axisC, axisC1 * axisH * axisW, axisNo * axisNi, axisC0};
  }

  if (srcFormat == "NDHWC" && dstFormat == "NDC1HWC0") {
    if (inShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size is not correct!");
      return false;
    }
    realSrcFormat = "NDHC";
    realDstFormat = "NDCHT";
    inShapeNew.push_back(inShape[0]);
    inShapeNew.push_back(inShape[1]);
    inShapeNew.push_back(inShape[3] * inShape[2]);
    inShapeNew.push_back(inShape[4]);
    int64_t cIdx = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
    int64_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
    int64_t axisC0 = c0Len;
    outShapeNew.push_back(inShape[0]);
    outShapeNew.push_back(inShape[1]);
    outShapeNew.push_back(axisC1);
    outShapeNew.push_back(inShape[3] * inShape[2]);
    outShapeNew.push_back(axisC0);
  }

  if (srcFormat == "NDC1HWC0" && dstFormat == "NCDHW") {
    if (inShape.size() != 6 || outShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    realSrcFormat = "NDCHT";
    realDstFormat = "NCDH";
    int64_t axisN = outShape[0];
    int64_t axisC = outShape[1];
    int64_t axisD = outShape[2];
    int64_t axisH = outShape[3];
    int64_t axisW = outShape[4];
    int64_t axisC0 = inShape[inShape.size() - 1];
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    inShapeNew = {axisN, axisD, axisC1, axisH * axisW, axisC0};
    outShapeNew = {axisN, axisC, axisD, axisH * axisW};
  }

  if (srcFormat == "FRACTAL_Z_3D" && dstFormat == "NCDHW") {
    if (inShape.size() < 2 || outShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    realSrcFormat = "DCHNT";
    realDstFormat = "NCDH";
    int64_t axisN = outShape[0];
    int64_t axisC = outShape[1];
    int64_t axisD = outShape[2];
    int64_t axisH = outShape[3];
    int64_t axisW = outShape[4];
    int64_t axisC0 = inShape[inShape.size() - 1];
    int64_t axisNi = inShape[inShape.size() - 2];
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    int64_t axisNo = GetCeilDiv(axisN, axisNi);
    inShapeNew = {axisD, axisC1, axisH * axisW, axisNo * axisNi, axisC0};
    outShapeNew = {axisN, axisC, axisD, axisH * axisW};
  }

  if (srcFormat == "NC1HWC0" && dstFormat == "NHWC") {
    if (inShape.size() != 5 || outShape.size() != 4) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size should be 5 and output's should be 4!");
      return false;
    }
    realSrcFormat = "NCHT";
    realDstFormat = "NHC";
    int64_t axisN = inShape[0];
    int64_t axisC1 = inShape[1];
    int64_t axisH = inShape[2];
    int64_t axisW = inShape[3];
    int64_t axisC0 = inShape[4];
    int64_t axisC = outShape[outShape.size() - 1];
    int64_t axisHW = axisH * axisW;
    inShapeNew = {axisN, axisC1, axisHW, axisC0};
    outShapeNew = {axisN, axisHW, axisC};
  }

  if (srcFormat == "NDC1HWC0" && dstFormat == "NDHWC") {
    if (inShape.size() != 6 || outShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The input shape dimension size should be 6 and output's should be 5!");
      return false;
    }
    realSrcFormat = "NCHT";
    realDstFormat = "NHC";
    int64_t axisN = inShape[0];
    int64_t axisD = inShape[1];
    int64_t axisC1 = inShape[2];
    int64_t axisH = inShape[3];
    int64_t axisW = inShape[4];
    int64_t axisC0 = inShape[5];
    int64_t axisC = outShape[outShape.size() - 1];
    int64_t axisHW = axisH * axisW;
    inShapeNew = {axisN * axisD, axisC1, axisHW, axisC0};
    outShapeNew = {axisN * axisD, axisHW, axisC};
  }

  if ((srcFormat == "FRACTAL_NZ") && (dstFormat == "ND" || dstFormat == "NCHW" || dstFormat == "NHWC")) {
    if (outShape.size() == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("trans_data", "The output shape dimension size cannot be 0!");
      return false;
    }
    realSrcFormat = "HCNT";
    realDstFormat = "HNC";
    int64_t axisN;
    int64_t axisH;
    int64_t axisC;
    if (outShape.size() == 1) {
      axisH = 1;
      axisN = 1;
      axisC = outShape[0];
    } else if (outShape.size() == 2) {
      axisH = 1;
      axisN = outShape[0];
      axisC = outShape[1];
    } else {
      int64_t shapeSize = 1;
      for (size_t i = 0; i < outShape.size() - 2; i++) {
        shapeSize *= outShape[i];
      }
      axisH = shapeSize;
      axisN = outShape[outShape.size() - 2];
      axisC = outShape[outShape.size() - 1];
    }
    int64_t axisC0 = inShape[inShape.size() - 1];
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    int64_t axisNo = GetCeilDiv(axisN, NI_16);
    inShapeNew = {axisH, axisC1, axisNo * NI_16, axisC0};
    outShapeNew = {axisH, axisN, axisC};
  }

  if (srcFormat == "FRACTAL_Z_3D" && dstFormat == "NDHWC") {
    if (inShape.size() != 4 || outShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    realSrcFormat = "DCHNT";
    realDstFormat = "NDHC";
    int64_t axisDC1HW = inShape[0];
    int64_t axisNo = inShape[1];
    int64_t axisNi = inShape[2];
    int64_t axisC0 = inShape[3];
    int64_t axisN = outShape[0];
    int64_t axisD = outShape[1];
    int64_t axisH = outShape[2];
    int64_t axisW = outShape[3];
    int64_t axisC = outShape[4];
    int64_t axisC1 = axisDC1HW / (axisD * axisH * axisW);

    inShapeNew = {axisD, axisC1, axisH * axisW, axisNo * axisNi, axisC0};
    outShapeNew = {axisN, axisD, axisH * axisW, axisC};
  }

  if (srcFormat == "FRACTAL_NZ" && dstFormat == "NC1HWC0") {
    if (outShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    realSrcFormat = "DCHNT";
    realDstFormat = "NDHC";

    int64_t axisD = 1;
    int64_t axisC = 1;
    int64_t axisN = outShape[0];
    int64_t axisC1 = outShape[1];
    int64_t axisH = outShape[2];
    int64_t axisW = outShape[3];
    int64_t axisC0 = outShape[4];
    int64_t axisNo = GetCeilDiv(axisN, NI_16);
    int64_t axisNi = NI_16;

    inShapeNew = {axisD, axisC, axisC1 * axisH * axisW, axisNo * axisNi, axisC0};
    outShapeNew = {axisN, axisD, axisC1 * axisH * axisW, axisC0};
  }

  if ((srcFormat == "FRACTAL_Z" || srcFormat == "FRACTAL_ZN") && (dstFormat == "HWCN")) {
    realSrcFormat = "CHNT";
    realDstFormat = "HCN";

    if (outShape.size() != 4 || inShape.size() < 2) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect!");
      return false;
    }

    int64_t axisH = outShape[0];
    int64_t axisW = outShape[1];
    int64_t axisC = outShape[2];
    int64_t axisN = outShape[3];
    int64_t axisC0 = inShape[inShape.size() - 1];
    int64_t axisNi = inShape[inShape.size() - 2];
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    int64_t axisNo = GetCeilDiv(axisN, axisNi);

    inShapeNew = {axisC1, axisH * axisW, axisNo * axisNi, axisC0};
    outShapeNew = {axisH * axisW, axisC, axisN};
  }

  if (srcFormat == "FRACTAL_Z_3D" && dstFormat == "DHWCN") {
    if (inShape.size() < 2 || outShape.size() != 5) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect");
      return false;
    }
    realSrcFormat = "DCHNT";
    realDstFormat = "DHCN";
    int64_t axisD = outShape[0];
    int64_t axisH = outShape[1];
    int64_t axisW = outShape[2];
    int64_t axisC = outShape[3];
    int64_t axisN = outShape[4];
    int64_t axisC0 = inShape[inShape.size() - 1];
    int64_t axisNi = inShape[inShape.size() - 2];
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    int64_t axisNo = GetCeilDiv(axisN, axisNi);
    inShapeNew = {axisD, axisC1, axisH * axisW, axisNo * axisNi, axisC0};
    outShapeNew = {axisD, axisH * axisW, axisC, axisN};
  }

  if ((srcFormat == "FRACTAL_Z" || srcFormat == "FRACTAL_ZN") && (dstFormat == "NCHW")) {
    realSrcFormat = "CHNT";
    realDstFormat = "NCH";

    if (outShape.size() != 4 || inShape.size() < 2) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect!");
      return false;
    }

    int64_t axisN = outShape[0];
    int64_t axisC = outShape[1];
    int64_t axisH = outShape[2];
    int64_t axisW = outShape[3];
    int64_t axisC0 = inShape[inShape.size() - 1];
    int64_t axisNi = inShape[inShape.size() - 2];
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    int64_t axisNo = GetCeilDiv(axisN, axisNi);

    inShapeNew = {axisC1, axisH * axisW, axisNo * axisNi, axisC0};
    outShapeNew = {axisN, axisC, axisH * axisW};
  }

  if ((srcFormat == "FRACTAL_Z" || srcFormat == "FRACTAL_ZN") && (dstFormat == "ND")) {
    realSrcFormat = "HCNT";
    realDstFormat = "HCN";

    if (inShape.size() < 2) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetRenew2Shape error, shape size incorrect!");
      return false;
    }
    int64_t axisN;
    int64_t axisH;
    int64_t axisC;
    if (outShape.size() == 1) {
      axisH = 1;
      axisC = 1;
      axisN = outShape[0];
    } else if (outShape.size() == 2) {
      axisH = 1;
      axisC = outShape[0];
      axisN = outShape[1];
    } else {
      int64_t shapeSize = 1;
      for (size_t i = 0; i < outShape.size() - 2; i++) {
        shapeSize *= outShape[i];
      }
      axisH = shapeSize;
      axisC = outShape[outShape.size() - 2];
      axisN = outShape[outShape.size() - 1];
    }
    int64_t axisC0 = inShape[inShape.size() - 1];
    int64_t axisNi = inShape[inShape.size() - 2];
    int64_t axisC1 = GetCeilDiv(axisC, axisC0);
    int64_t axisNo = GetCeilDiv(axisN, axisNi);

    inShapeNew = {axisH, axisC1, axisNo * axisNi, axisC0};
    outShapeNew = {axisH, axisC, axisN};
  }

  return true;
}

int32_t GetMultiCoreAxis(std::vector<int64_t> inShape, int32_t axisPosC, int64_t blockElemCnt, int64_t c0Len,
                         int64_t coreNum) {
  int32_t shapeLen = inShape.size();
  bool axisCNotLastDim = axisPosC + 1 != shapeLen;
  std::vector<int32_t> coreLpCnt;

  for (int32_t index = 0; index < shapeLen; index++) {
    int32_t tmpFullCycleLoopCnt;
    int32_t leftLoopCnt;
    int32_t fullCycleLoopCnt;
    if (index + 1 == shapeLen) {
      if (GetFloorDiv(inShape[index], 8 * blockElemCnt * coreNum) > 0) {
        tmpFullCycleLoopCnt = coreNum;
      } else {
        tmpFullCycleLoopCnt = 0;
      }
      leftLoopCnt = GetCeilDiv(inShape[index], 8 * blockElemCnt) % coreNum;
    } else if (index == axisPosC && axisCNotLastDim) {
      if (GetFloorDiv(inShape[index], c0Len * coreNum) > 0) {
        tmpFullCycleLoopCnt = coreNum;
      } else {
        tmpFullCycleLoopCnt = 0;
      }
      leftLoopCnt = GetCeilDiv(inShape[index], c0Len) % coreNum;
    } else {
      if (GetFloorDiv(inShape[index], coreNum) > 0) {
        tmpFullCycleLoopCnt = coreNum;
      } else {
        tmpFullCycleLoopCnt = 0;
      }
      leftLoopCnt = inShape[index] % coreNum;
    }

    if (tmpFullCycleLoopCnt > 0 && leftLoopCnt == 0) {
      fullCycleLoopCnt = 2 * tmpFullCycleLoopCnt;
    } else {
      fullCycleLoopCnt = tmpFullCycleLoopCnt;
    }
    coreLpCnt.push_back(fullCycleLoopCnt + leftLoopCnt);
  }

  return max_element(coreLpCnt.begin(), coreLpCnt.end()) - coreLpCnt.begin();
}

bool IsDoWithTransposeFormats(const std::string& srcFormat, const std::string& dstFormat){
  const std::vector<std::string> FormatList = {"NCHW", "NHWC", "HWCN", "CHWN"};
  if (std::find(FormatList.begin(), FormatList.end(), srcFormat) != FormatList.end() &&
      std::find(FormatList.begin(), FormatList.end(), dstFormat) != FormatList.end() && dstFormat != srcFormat) {
    return true;
  } else {
    return false;
  }
}

bool IsDoWithPositiveSourceNtc100(const std::string& srcFormat, const std::string& dstFormat) {
  const std::vector<std::pair<std::string, std::string>> supportSrcDstFormats = {
      {"NCDHW", "NDC1HWC0"},
      {"NCHW", "NC1HWC0"},
      {"HWCN", "FRACTAL_Z"},
      {"HWCN", "FRACTAL_ZN"},
      {"DHWCN", "FRACTAL_Z_3D"},
      {"NCDHW", "FRACTAL_Z_3D"},
      {"ND", "FRACTAL_Z"},
      {"ND", "FRACTAL_ZN"},
      {"NCHW", "FRACTAL_Z"},
      {"NCHW", "FRACTAL_ZN"}};
  return std::find(supportSrcDstFormats.begin(), supportSrcDstFormats.end(),
                   std::pair<std::string, std::string>(srcFormat, dstFormat)) != supportSrcDstFormats.end();
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool TransDataTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                     OpRunInfo& runInfo) {
  PROFILING_TILING_INIT(opType.c_str());
  OP_LOGI(opType.c_str(), "Tiling is running.");
  if (op_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op TransDataTiling: op_info json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < 1 || opParas.inputs[0].tensor.empty()) {

    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op TransDataTiling: input shape error.");
    return false;
  }
  if (opParas.outputs.empty() || opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op TransDataTiling: output shape error.");
    return false;
  }

  std::string srcFormat = opParas.inputs[0].tensor[0].format;
  std::string dstFormat = opParas.outputs[0].tensor[0].format;
  OP_LOGD(opType, "Input format is [%s], Output format is [%s].",
          srcFormat.c_str(), dstFormat.c_str());
  if (IsDoWithTransposeFormats(srcFormat, dstFormat)) {
    std::vector<int64_t> perm_shape;
    TeOpParas opParasTranspose;
    opParasTranspose = opParas;
    perm_shape.push_back(4);
    ge::Shape ge_shape(perm_shape);
    ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge::Format::FORMAT_ND, ge::DataType::DT_INT64));
    int64_t buf[4];
    std::map<char, std::int64_t> DimIndex;
    for (size_t i = 0; i < srcFormat.size(); i++) {
      DimIndex[srcFormat[i]] = i;
    }
    for (size_t i = 0; i < dstFormat.size(); i++) {
      buf[i] = DimIndex[dstFormat[i]];
    }
    opParasTranspose.const_inputs["perm"] = std::make_tuple((const unsigned char*)buf, sizeof(buf), const_tensor);
    bool ret = TransposeTiling(opType, opParasTranspose, op_info, runInfo);
    return ret;
  }
  std::vector<int64_t> inShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> outShape = opParas.outputs[0].tensor[0].shape;
  std::string realSrcFormat;
  std::string realDstFormat;
  std::string dType;
  int64_t ubSize = 0;
  int64_t blockDim = 0;
  std::vector<int64_t> inShapeNew;
  std::vector<int64_t> outShapeNew;
  std::vector<int64_t> combAxis;
  int64_t inputSize = 0;
  int64_t hiddenSize = 0;
  int64_t group = 1;

  // get the point ts after get info from opParas
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  bool flag = GetCompileParams(op_info, srcFormat, dstFormat, dType, ubSize, blockDim, inputSize, hiddenSize, group,
                               opType);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransDataTiling: GetCompileParams error.");
    return false;
  }

  // get the point ts after get compile info
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  int64_t c0Len = GetC0Len(dType);
  bool ret = CheckTensorShape(opType, ubSize, blockDim, outShape);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op TransDataTiling: CheckTensor Failed.");
    return ret;
  }
  int64_t blockElemCnt = BLOCK_BYTE_SIZE / GetDTypeLen(dType);

  flag = GetRenew2Shape(inShape, outShape, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                          outShapeNew, realSrcFormat, realDstFormat);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransDataTiling: GetRenew2Shape tiling params error");
    return false;
  }

  if (IsDoWithPositiveSourceNtc100(srcFormat, dstFormat)) {
    TransDataNtc100Param runParams100;
    flag = TilingPositiveSourceNtc100(inShape, outShape, srcFormat, dstFormat, blockDim,
                                      blockElemCnt, ubSize, c0Len, dType, runParams100);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransDataTiling: get TilingPositiveSourceNtc100 tiling params error");
      return false;
    }
    SetRunningNtc100Params(runParams100, runInfo);
    OP_LOGD(opType.c_str(), "start print tiling parameters in ntc 100: %s", runParams100.to_string().c_str());
  } else if (realSrcFormat[realSrcFormat.length() - 1] == 'C' && realDstFormat[realDstFormat.length() - 1] == 'T') {
    if (realSrcFormat[realSrcFormat.length() - 2] == realDstFormat[realDstFormat.length() - 2]) {
      TransDataMode1010Param runParamsPart1;
      flag = TillingPositiveMode1010(inShapeNew, outShapeNew, realSrcFormat, realDstFormat,
                                     blockDim, blockElemCnt, ubSize, runParamsPart1);
      if (!flag) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransDataTiling: get TransDataMode101Param tiling params error");
        return false;
      }
      SetRunningMode1010Params(runParamsPart1, runInfo);
      OP_LOGD(opType.c_str(), "start print runParams");
      PrintTilingMode1010Params(opType, runParamsPart1);
    } else {
      TransDataMode1011Param runParamsPart1;
      flag = TillingPositiveMode1011(inShapeNew, outShapeNew, realSrcFormat, realDstFormat,
                                     blockDim, blockElemCnt, ubSize, runParamsPart1);
      if (!flag) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransDataTiling: get TransDataMode101Param tiling params error");
        return false;
      }
      SetRunningMode1011Params(runParamsPart1, runInfo);
      OP_LOGD(opType.c_str(), "start print runParams");
      PrintTilingMode1011Params(opType, runParamsPart1);
    }

  } else if ((srcFormat == "NC1HWC0" && dstFormat == "NHWC") || (srcFormat == "FRACTAL_NZ" && dstFormat == "ND") ||
              (srcFormat == "FRACTAL_Z_3D" && dstFormat == "NDHWC") ||
              (srcFormat == "FRACTAL_NZ" && (dstFormat == "NCHW" || dstFormat == "NHWC" || dstFormat == "NC1HWC0")) ||
              (srcFormat == "NDC1HWC0" && dstFormat == "NDHWC")) {
    TransDataTc201Param runParams201;
    flag = TilingNegativeTc201(inShapeNew, outShapeNew, realSrcFormat, realDstFormat, blockDim, blockElemCnt, dType,
                               ubSize, runParams201);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransDataTiling: get TilingNegativeTc201 tiling params error");
      return false;
    }
    OP_LOGD(opType.c_str(), "***start to put mode 201 tiling parameters");
    SetRunningTc201Params(runParams201, runInfo);
    PrintTilingModeTc201Params(opType, runParams201);
  } else if ((srcFormat == "NC1HWC0" && dstFormat == "NCHW") ||
             (srcFormat == "FRACTAL_Z_3D" && dstFormat == "NCDHW") ||
             (srcFormat == "NDC1HWC0" && dstFormat == "NCDHW") ||
             ((srcFormat == "FRACTAL_Z" || srcFormat == "FRACTAL_ZN") && (dstFormat == "HWCN")) ||
             ((srcFormat == "FRACTAL_Z" || srcFormat == "FRACTAL_ZN") && (dstFormat == "NCHW")) ||
             ((srcFormat == "FRACTAL_Z" || srcFormat == "FRACTAL_ZN") && (dstFormat == "ND")) ||
             (srcFormat == "FRACTAL_Z_3D" && dstFormat == "DHWCN")) {
    TransDataNtc200Param runParams200;
    flag = TilingNegativeNtc200(inShapeNew, outShapeNew, realSrcFormat, realDstFormat, blockDim, blockElemCnt, dType,
                                ubSize, runParams200);
    if (!flag) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransDataTiling: get TilingNegativeNtc200 tiling params error");
      return false;
    }
    SetRunningNtc200Params(runParams200, runInfo);
    OP_LOGD(opType.c_str(), "start print tiling parameters in mode 200");
    PrintTilingModeNtc200Params(opType, runParams200);
  }

  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  // block_dim, core num used in tik op
  runInfo.block_dim = blockDim;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "tiling run success.");

  // get the point ts and calcu the all time cost
  PROFILING_TILING_END()
  return true;
}

// register tiling interface of the TransData op
REGISTER_OP_TILING_FUNC_BUFFERED(TransData, TransDataTiling);

}  // namespace optiling

