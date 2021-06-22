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
 * \file transpose_d.cpp
 * \brief dynamic TransposeD op tiling
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int64_t BLOCK_SIZE = 32;
const int64_t VNC_LINE_NUM = 16;
const std::vector<int64_t> PERM_0_1 = {0, 1};
const std::vector<int64_t> PERM_1_0 = {1, 0};
const std::vector<int64_t> PERM_0_1_2 = {0, 1, 2};
const std::vector<int64_t> PERM_0_2_1 = {0, 2, 1};
const std::vector<int64_t> PERM_1_0_2 = {1, 0, 2};
const std::vector<int64_t> PERM_1_2_0 = {1, 2, 0};
const std::vector<int64_t> PERM_2_0_1 = {2, 0, 1};
const std::vector<int64_t> PERM_2_1_0 = {2, 1, 0};

int64_t GetFloorDiv(const int64_t uValue, const int64_t dValue) {
  int64_t resValue = 0;

  if (dValue == 0) {
      return uValue;
  }
  resValue = uValue / dValue;

  return resValue;
}

int64_t GetCeilDiv(const int64_t uValue, const int64_t dValue) {
  int64_t resValue = 0;

  if (dValue == 0) {
      return uValue;
  }
  resValue = (uValue + dValue - 1) / dValue;

  return resValue;
}

int64_t GetMul(const int64_t lValue, const int64_t rValue) {
  int64_t resValue = 0;

  resValue = lValue * rValue;

  return resValue;
}

int64_t GetMod(const int64_t lValue, const int64_t rValue) {
  int64_t resValue = 0;

  if (rValue == 0) {
      return lValue;
  }
  resValue = lValue % rValue;

  return resValue;
}

int64_t GetSub(const int64_t lValue, const int64_t rValue) {
  int64_t resValue = 0;

  resValue = lValue - rValue;

  return resValue;
}

int64_t GetMax(const int64_t lValue, const int64_t rValue) {
  int64_t resValue = 0;

  if (lValue > rValue) {
    resValue = lValue;
  } else {
    resValue = rValue;
  }

  return resValue;
}

int64_t GetMin(const int64_t lValue, const int64_t rValue) {
  int64_t resValue = 0;

  if (lValue < rValue) {
    resValue = lValue;
  } else {
    resValue = rValue;
  }

  return resValue;
}

bool CheckTensorShape(const std::string& opType, const TeOpParas& opParas, std::vector<int64_t>& inPerm) {
  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [TransposeD] TransposeDTiling: input shape error");
    return false;
  }

  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [TransposeD] TransposeDTiling: output shape error");
    return false;
  }

  std::vector<int64_t> inShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> outShape = opParas.outputs[0].tensor[0].shape;
  int64_t inDims = inShape.size();
  int64_t outDims = outShape.size();
  int64_t permDims = inPerm.size();

  if (inDims == 0 || inDims != outDims || inDims != permDims) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the dim of inputs is invalid.");
    return false;
  }

  for (int32_t i = 0; i < inDims; i++) {
    if (inShape[inPerm[i]] != outShape[i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the dim of inputs or outputs is conflict with perm.");
      return false;
    }
  }

  return true;
}

bool GetCompileParams(const nlohmann::json& opCompileInfoJson, int64_t& coreNum, int64_t& ubSize,
                      std::vector<int64_t>& inPerm, std::string& dType) {
  using namespace nlohmann;

  auto allVars = opCompileInfoJson["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransposeDTiling", "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransposeDTiling", "GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();

  if (allVars.count("perm") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransposeDTiling", "GetCompileParams, get perm error");
    return false;
  }
  inPerm = allVars["perm"].get<std::vector<int64_t>>();

  if (allVars.count("dtype") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransposeDTiling", "GetCompileParams, get dtype error");
    return false;
  }
  dType = allVars["dtype"].get<std::string>();

  GELOGD("op [TransposeDTiling] : GetCompileParams, coreNum[%d], ubSize[%d], dType[%s], inPerm size[%d]:(%d, %d).",
         coreNum, ubSize, dType.c_str(), inPerm.size(), inPerm[0], inPerm[1]);

  return true;
}

/*
 * needCoreNum
 * ubOffset
 * cAxisMaxLen
 * perCoreCAxisSize
 * perCoreCAxisLoopCnt
 * perCoreCAxisLeftSize
 * lastCoreBAixSize
 * lastCoreCAxisLoopCnt
 * lastCoreCAxisLeftSize
 * bAxisMaxLen
 * bAxisLoopCnt
 * bAxisLeftSize
 */
bool Float32Axis1Axis0Tiling(int64_t& dBytes, int64_t& coreNum, int64_t& ubSize, int64_t& bAxis, int64_t& cAxis,
                             int64_t& needCoreNum, int64_t& ubOffset, int64_t& cAxisMaxLen, int64_t& perCoreCAxisSize,
                             int64_t& perCoreCAxisLoopCnt, int64_t& perCoreCAxisLeftSize, int64_t& lastCoreCAxisSize,
                             int64_t& lastCoreCAxisLoopCnt, int64_t& lastCoreCAxisLeftSize, int64_t& bAxisMaxLen,
                             int64_t& bAxisLoopCnt, int64_t& bAxisLeftSize) {
  int64_t ubPart = 2;
  int64_t dataOneBlock = GetFloorDiv(BLOCK_SIZE, dBytes);
  int64_t vncLineSize = 0;

  ubOffset = GetFloorDiv(ubSize, ubPart);  // unit is element, split into two parts
  // to make sure the line size is block align
  vncLineSize = GetMul(GetFloorDiv(GetFloorDiv(ubOffset, VNC_LINE_NUM), dataOneBlock), dataOneBlock);
  // get need core num
  needCoreNum = GetCeilDiv(bAxis, GetCeilDiv(bAxis, coreNum));
  // set to 16 directly
  bAxisMaxLen = 16;
  bAxisLoopCnt = GetFloorDiv(bAxis, bAxisMaxLen);
  bAxisLeftSize = GetMod(bAxis, bAxisMaxLen);
  // to make sure each loop process data size is block align
  cAxisMaxLen = GetMul(GetFloorDiv(GetFloorDiv(vncLineSize, bAxisMaxLen), dataOneBlock), dataOneBlock);
  // to make sure needCoreNum-1 cores process data size is block align
  perCoreCAxisSize = GetMul(GetFloorDiv(GetCeilDiv(cAxis, needCoreNum), dataOneBlock), dataOneBlock);
  perCoreCAxisLoopCnt = GetFloorDiv(perCoreCAxisSize, cAxisMaxLen);
  perCoreCAxisLeftSize = GetMod(perCoreCAxisSize, cAxisMaxLen);
  // last core process data size
  lastCoreCAxisSize = GetSub(cAxis, GetMul(perCoreCAxisSize, GetSub(needCoreNum, 1)));
  lastCoreCAxisLoopCnt = GetFloorDiv(lastCoreCAxisSize, cAxisMaxLen);
  lastCoreCAxisLeftSize = GetMod(lastCoreCAxisSize, cAxisMaxLen);

  return true;
}

/*
 * needCoreNum
 * ubOffset
 * cAxisMaxLen
 * perCoreCAxisSize
 * perCoreCAxisLoopCnt
 * perCoreCAxisLeftSize
 * lastCoreBAixSize
 * lastCoreCAxisLoopCnt
 * lastCoreCAxisLeftSize
 * bAxisMaxLen
 * bAxisLoopCnt
 * bAxisLeftSize
 */
bool Float32Axis1Axis0Axis2Tiling(int64_t& dBytes, int64_t& coreNum, int64_t& ubSize, int64_t& aAxis, int64_t& bAxis,
                                  int64_t& cAxis, int64_t& needCoreNum, int64_t& ubOffset, int64_t& cAxisMaxLen,
                                  int64_t& perCoreCAxisSize, int64_t& perCoreCAxisLoopCnt,
                                  int64_t& perCoreCAxisLeftSize, int64_t& lastCoreCAxisSize,
                                  int64_t& lastCoreCAxisLoopCnt, int64_t& lastCoreCAxisLeftSize, int64_t& bAxisMaxLen,
                                  int64_t& bAxisLoopCnt, int64_t& bAxisLeftSize, int64_t& tilingMode) {
  int64_t ubPart = 1;
  int64_t dataOneBlock = GetFloorDiv(BLOCK_SIZE, dBytes);
  int64_t vncLineSize = 0;
  int64_t maxCoreCAxisSize = 0;
  int64_t lastCoreCAxisBlockAlign = 0;
  int64_t bAxisBlockAlign = 0;

  ubOffset = 0;
  if (cAxis < dataOneBlock) {
    ubPart = 2;
    tilingMode = 1023;
    ubOffset = GetFloorDiv(ubSize, ubPart);  // unit is element, split into two parts
    // get need core num
    needCoreNum = GetCeilDiv(bAxis, GetCeilDiv(bAxis, coreNum));
    // to make sure the line size is block align
    vncLineSize = GetMul(GetFloorDiv(GetFloorDiv(ubOffset, VNC_LINE_NUM), dataOneBlock), dataOneBlock);
    // the last dim size is less than one block size
    bAxisMaxLen = GetFloorDiv(vncLineSize, dataOneBlock);
    bAxisLoopCnt = GetFloorDiv(aAxis, bAxisMaxLen);
    bAxisLeftSize = GetMod(aAxis, bAxisMaxLen);
    // to make sure each loop process data size is block align
    if (bAxisLoopCnt > 0) {
      cAxisMaxLen = 1;
    } else {
      bAxisBlockAlign = GetMul(aAxis, dataOneBlock);
      cAxisMaxLen = GetFloorDiv(vncLineSize, bAxisBlockAlign);
    }
    perCoreCAxisSize = GetMul(GetFloorDiv(GetCeilDiv(bAxis, needCoreNum), dataOneBlock), dataOneBlock);
    perCoreCAxisLoopCnt = GetFloorDiv(perCoreCAxisSize, cAxisMaxLen);
    perCoreCAxisLeftSize = GetMod(perCoreCAxisSize, cAxisMaxLen);
    // last core process data size
    lastCoreCAxisSize = GetSub(bAxis, GetMul(perCoreCAxisSize, GetSub(needCoreNum, 1)));
    lastCoreCAxisLoopCnt = GetFloorDiv(lastCoreCAxisSize, cAxisMaxLen);
    lastCoreCAxisLeftSize = GetMod(lastCoreCAxisSize, cAxisMaxLen);

  } else if (cAxis >= bAxis) {
    tilingMode = 1022;
    // get need core num
    needCoreNum = GetCeilDiv(cAxis, GetCeilDiv(cAxis, coreNum));
    // get multiple core parameters, to make sure needCoreNum-1 cores process data size is block align
    cAxisMaxLen = ubSize;
    perCoreCAxisSize = GetMul(GetFloorDiv(GetCeilDiv(cAxis, needCoreNum), dataOneBlock), dataOneBlock);
    perCoreCAxisLoopCnt = GetFloorDiv(perCoreCAxisSize, cAxisMaxLen);
    perCoreCAxisLeftSize = GetMod(perCoreCAxisSize, cAxisMaxLen);
    // last core process data size
    lastCoreCAxisSize = GetSub(cAxis, GetMul(perCoreCAxisSize, GetSub(needCoreNum, 1)));
    lastCoreCAxisLoopCnt = GetFloorDiv(lastCoreCAxisSize, cAxisMaxLen);
    lastCoreCAxisLeftSize = GetMod(lastCoreCAxisSize, cAxisMaxLen);
    // get no core parameters
    if (perCoreCAxisLoopCnt > 0 || lastCoreCAxisLoopCnt > 0) {
      bAxisMaxLen = 1;
    } else {
      lastCoreCAxisBlockAlign = GetMul(GetCeilDiv(lastCoreCAxisSize, dataOneBlock), dataOneBlock);
      maxCoreCAxisSize = GetMax(perCoreCAxisSize, lastCoreCAxisBlockAlign);
      bAxisMaxLen = GetFloorDiv(ubSize, maxCoreCAxisSize);
    }
    bAxisLoopCnt = GetFloorDiv(bAxis, bAxisMaxLen);
    bAxisLeftSize = GetMod(bAxis, bAxisMaxLen);

  } else {
    tilingMode = 1021;
    // get need core num
    needCoreNum = GetCeilDiv(bAxis, GetCeilDiv(bAxis, coreNum));
    // get no core parameters
    bAxisMaxLen = ubSize;
    bAxisLoopCnt = GetFloorDiv(cAxis, bAxisMaxLen);
    bAxisLeftSize = GetMod(cAxis, bAxisMaxLen);
    // get multiple core parameters
    // to make sure needCoreNum-1 cores process data size is block align
    perCoreCAxisSize = GetMul(GetFloorDiv(GetCeilDiv(bAxis, needCoreNum), dataOneBlock), dataOneBlock);
    lastCoreCAxisSize = GetSub(bAxis, GetMul(perCoreCAxisSize, GetSub(needCoreNum, 1)));
    if (bAxisLoopCnt > 0) {
      cAxisMaxLen = 1;
    } else {
      bAxisBlockAlign = GetMul(GetCeilDiv(cAxis, dataOneBlock), dataOneBlock);
      cAxisMaxLen = GetFloorDiv(ubSize, bAxisBlockAlign);
    }
    perCoreCAxisLoopCnt = GetFloorDiv(perCoreCAxisSize, cAxisMaxLen);
    perCoreCAxisLeftSize = GetMod(perCoreCAxisSize, cAxisMaxLen);
    // last core process data size
    lastCoreCAxisLoopCnt = GetFloorDiv(lastCoreCAxisSize, cAxisMaxLen);
    lastCoreCAxisLeftSize = GetMod(lastCoreCAxisSize, cAxisMaxLen);
  }

  return true;
}

/*
 * needCoreNum
 * cAxisMaxLen
 * perCoreCAxisSize
 * perCoreCAxisLoopCnt
 * perCoreCAxisLeftSize
 * lastCoreBAixSize
 * lastCoreCAxisLoopCnt
 * lastCoreCAxisLeftSize
 */
bool Float32Axis0Axis1Tiling(int64_t& dBytes, int64_t& coreNum, int64_t& ubSize, int64_t& totalHWSize,
                             int64_t& needCoreNum, int64_t& cAxisMaxLen, int64_t& perCoreCAxisSize,
                             int64_t& perCoreCAxisLoopCnt, int64_t& perCoreCAxisLeftSize, int64_t& lastCoreCAxisSize,
                             int64_t& lastCoreCAxisLoopCnt, int64_t& lastCoreCAxisLeftSize) {
  int64_t dataOneBlock = GetFloorDiv(BLOCK_SIZE, dBytes);

  // get need core num
  needCoreNum = GetCeilDiv(totalHWSize, GetCeilDiv(totalHWSize, coreNum));
  // to make sure each loop process data size is block align
  cAxisMaxLen = GetMul(GetFloorDiv(ubSize, dataOneBlock), dataOneBlock);
  // to make sure needCoreNum-1 cores process data size is block align
  perCoreCAxisSize = GetMul(GetFloorDiv(GetCeilDiv(totalHWSize, needCoreNum), dataOneBlock), dataOneBlock);
  perCoreCAxisLoopCnt = GetFloorDiv(perCoreCAxisSize, cAxisMaxLen);
  perCoreCAxisLeftSize = GetMod(perCoreCAxisSize, cAxisMaxLen);
  // last core process data size
  lastCoreCAxisSize = GetSub(totalHWSize, GetMul(perCoreCAxisSize, GetSub(needCoreNum, 1)));
  lastCoreCAxisLoopCnt = GetFloorDiv(lastCoreCAxisSize, cAxisMaxLen);
  lastCoreCAxisLeftSize = GetMod(lastCoreCAxisSize, cAxisMaxLen);

  return true;
}

/*
 * needCoreNum
 * bAxisMaxLen
 * bAxisLoopCnt
 * bAxisLeftSize
 */
bool Float32Axis2Axis1Axis0Tiling(int64_t& dBytes, int64_t& coreNum, int64_t& ubSize, int64_t& aAxis,
                                  int64_t& needCoreNum, int64_t& bAxisMaxLen, int64_t& bAxisLoopCnt,
                                  int64_t& bAxisLeftSize) {
  int64_t dataOneBlock = GetFloorDiv(BLOCK_SIZE, dBytes);
  needCoreNum = 1;
  bAxisMaxLen = GetFloorDiv(ubSize, dataOneBlock);
  bAxisLoopCnt = GetFloorDiv(aAxis, bAxisMaxLen);
  bAxisLeftSize = GetMod(aAxis, bAxisMaxLen);

  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool TransposeDTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                      OpRunInfo& runInfo) {
  GELOGI("op[%s] tiling running.", opType.c_str());
  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [TransposeD] TransposeDTiling: input shape error");
    return false;
  }

  // get compile info
  int64_t ubSize = 0;
  int64_t coreNum = 0;
  int64_t dBytes = 4;
  int64_t tilingMode = 1;
  int64_t needCoreNum = 0;
  int64_t ubOffset = 0;
  int64_t cAxisMaxLen = 0;
  int64_t perCoreCAxisSize = 0;
  int64_t perCoreCAxisLoopCnt = 0;
  int64_t perCoreCAxisLeftSize = 0;
  int64_t lastCoreCAxisSize = 0;
  int64_t lastCoreCAxisLoopCnt = 0;
  int64_t lastCoreCAxisLeftSize = 0;
  int64_t bAxisMaxLen = 0;
  int64_t bAxisLoopCnt = 0;
  int64_t bAxisLeftSize = 0;
  int64_t totalHWSize = 0;
  std::vector<int64_t> inPerm;
  std::string dType;
  std::vector<int64_t> inShape = opParas.inputs[0].tensor[0].shape;
  int64_t aAxis = 1;
  int64_t bAxis = 1;
  int64_t cAxis = 1;

  bool flag = GetCompileParams(op_info, coreNum, ubSize, inPerm, dType);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: GetCompileParams error.");
    return false;
  }

  bool ret = CheckTensorShape(opType, opParas, inPerm);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: inputs or outputs shape are invalid.");
    return false;
  }

  if (inPerm == PERM_1_0) {
    if (inShape.size() != 2) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 2!");
      return false;
    }
    tilingMode = 101;
    bAxis = inShape[0];
    cAxis = inShape[1];
    bool ret =
        Float32Axis1Axis0Tiling(dBytes, coreNum, ubSize, bAxis, cAxis, needCoreNum, ubOffset, cAxisMaxLen,
                                perCoreCAxisSize, perCoreCAxisLoopCnt, perCoreCAxisLeftSize, lastCoreCAxisSize,
                                lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize, bAxisMaxLen, bAxisLoopCnt, bAxisLeftSize);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }

  } else if (inPerm == PERM_1_2_0) {
    if (inShape.size() != 3) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 3!");
      return false;
    }
    tilingMode = 1201;
    bAxis = inShape[0];
    cAxis = GetMul(inShape[1], inShape[2]);
    bool ret =
        Float32Axis1Axis0Tiling(dBytes, coreNum, ubSize, bAxis, cAxis, needCoreNum, ubOffset, cAxisMaxLen,
                                perCoreCAxisSize, perCoreCAxisLoopCnt, perCoreCAxisLeftSize, lastCoreCAxisSize,
                                lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize, bAxisMaxLen, bAxisLoopCnt, bAxisLeftSize);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }

  } else if (inPerm == PERM_2_0_1) {
    if (inShape.size() != 3) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 3!");
      return false;
    }
    tilingMode = 2011;
    bAxis = GetMul(inShape[0], inShape[1]);
    cAxis = inShape[2];
    bool ret =
        Float32Axis1Axis0Tiling(dBytes, coreNum, ubSize, bAxis, cAxis, needCoreNum, ubOffset, cAxisMaxLen,
                                perCoreCAxisSize, perCoreCAxisLoopCnt, perCoreCAxisLeftSize, lastCoreCAxisSize,
                                lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize, bAxisMaxLen, bAxisLoopCnt, bAxisLeftSize);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }

  } else if (inPerm == PERM_0_2_1) {
    if (inShape.size() != 3) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 3!");
      return false;
    }
    tilingMode = 211;
    aAxis = inShape[0];
    bAxis = inShape[1];
    cAxis = inShape[2];
    bool ret =
        Float32Axis1Axis0Tiling(dBytes, coreNum, ubSize, bAxis, cAxis, needCoreNum, ubOffset, cAxisMaxLen,
                                perCoreCAxisSize, perCoreCAxisLoopCnt, perCoreCAxisLeftSize, lastCoreCAxisSize,
                                lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize, bAxisMaxLen, bAxisLoopCnt, bAxisLeftSize);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }

  } else if (inPerm == PERM_0_1) {
    if (inShape.size() != 2) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 2!");
      return false;
    }
    tilingMode = 11;
    bAxis = inShape[0];
    cAxis = inShape[1];
    totalHWSize = GetMul(bAxis, cAxis);
    bool ret = Float32Axis0Axis1Tiling(dBytes, coreNum, ubSize, totalHWSize, needCoreNum, cAxisMaxLen, perCoreCAxisSize,
                                       perCoreCAxisLoopCnt, perCoreCAxisLeftSize, lastCoreCAxisSize,
                                       lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize);

    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }

  } else if (inPerm == PERM_0_1_2) {
    if (inShape.size() != 3) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 3!");
      return false;
    }
    tilingMode = 121;
    bAxis = inShape[0];
    cAxis = GetMul(inShape[1], inShape[2]);
    totalHWSize = GetMul(bAxis, cAxis);
    bool ret = Float32Axis0Axis1Tiling(dBytes, coreNum, ubSize, totalHWSize, needCoreNum, cAxisMaxLen, perCoreCAxisSize,
                                       perCoreCAxisLoopCnt, perCoreCAxisLeftSize, lastCoreCAxisSize,
                                       lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize);

    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }

  } else if (inPerm == PERM_1_0_2) {
    if (inShape.size() != 3) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 3!");
      return false;
    }
    aAxis = inShape[0];
    bAxis = inShape[1];
    cAxis = inShape[2];
    bool ret = Float32Axis1Axis0Axis2Tiling(dBytes, coreNum, ubSize, aAxis, bAxis, cAxis, needCoreNum, ubOffset,
                                            cAxisMaxLen, perCoreCAxisSize, perCoreCAxisLoopCnt, perCoreCAxisLeftSize,
                                            lastCoreCAxisSize, lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize, bAxisMaxLen,
                                            bAxisLoopCnt, bAxisLeftSize, tilingMode);

    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }

  } else if (inPerm == PERM_2_1_0) {
    if (inShape.size() != 3) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: the shape dimension should be 3!");
      return false;
    }
    tilingMode = 2101;
    aAxis = inShape[0];
    bAxis = inShape[1];
    cAxis = inShape[2];
    bool ret = Float32Axis2Axis1Axis0Tiling(dBytes, coreNum, ubSize, aAxis, needCoreNum, bAxisMaxLen, bAxisLoopCnt,
                                            bAxisLeftSize);

    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "TransposeDTiling: calculate tiling parameters failed!");
      return false;
    }
  }

  GELOGD("op [TransposeDTiling] : tilingMode=%d, needCoreNum=%d", tilingMode, needCoreNum);
  GELOGD("op [TransposeDTiling] : ubOffset=%d, bAxisMaxLen=%d, cAxisMaxLen=%d", ubOffset, bAxisMaxLen, cAxisMaxLen);
  GELOGD("op [TransposeDTiling] : perCoreCAxisSize=%d, perCoreCAxisLoopCnt=%d, perCoreCAxisLeftSize=%d",
         perCoreCAxisSize, perCoreCAxisLoopCnt, perCoreCAxisLeftSize);
  GELOGD("op [TransposeDTiling] : lastCoreCAxisSize=%d, lastCoreCAxisLoopCnt=%d, lastCoreCAxisLeftSize=%d",
         lastCoreCAxisSize, lastCoreCAxisLoopCnt, lastCoreCAxisLeftSize);
  GELOGD("op [TransposeDTiling] : bAxisLoopCnt=%d, bAxisLeftSize=%d", bAxisLoopCnt, bAxisLeftSize);
  GELOGD("op [TransposeDTiling] : aAxis=%d, bAxis=%d, cAxis=%d", aAxis, bAxis, cAxis);

  // set tiling data
  ByteBufferPut(runInfo.tiling_data, tilingMode);
  ByteBufferPut(runInfo.tiling_data, needCoreNum);
  ByteBufferPut(runInfo.tiling_data, ubOffset);
  ByteBufferPut(runInfo.tiling_data, bAxisMaxLen);
  ByteBufferPut(runInfo.tiling_data, cAxisMaxLen);
  ByteBufferPut(runInfo.tiling_data, perCoreCAxisSize);
  ByteBufferPut(runInfo.tiling_data, perCoreCAxisLoopCnt);
  ByteBufferPut(runInfo.tiling_data, perCoreCAxisLeftSize);
  ByteBufferPut(runInfo.tiling_data, lastCoreCAxisSize);
  ByteBufferPut(runInfo.tiling_data, lastCoreCAxisLoopCnt);
  ByteBufferPut(runInfo.tiling_data, lastCoreCAxisLeftSize);
  ByteBufferPut(runInfo.tiling_data, bAxisLoopCnt);
  ByteBufferPut(runInfo.tiling_data, bAxisLeftSize);
  ByteBufferPut(runInfo.tiling_data, aAxis);
  ByteBufferPut(runInfo.tiling_data, bAxis);
  ByteBufferPut(runInfo.tiling_data, cAxis);

  // block_dim, core num used in tik op
  runInfo.block_dim = needCoreNum;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the TransposeD op
REGISTER_OP_TILING_FUNC_BUFFERED(TransposeD, TransposeDTiling);

}  // namespace optiling
