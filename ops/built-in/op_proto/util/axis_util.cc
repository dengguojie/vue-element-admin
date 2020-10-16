/**
 * @file axis_util.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief get the axis value
 *
 * @version 1.0
 *
 */
#include "axis_util.h"
#include "framework/omg/omg_inner_types.h"

namespace ge {
AxisUtil::AxisUtil() {
  getAxisValueFuncMap = {
      {FORMAT_NCHW,
       std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNCHW)},
      {FORMAT_NHWC,
       std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNHWC)},
      {FORMAT_NC1HWC0,
       std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNC1HWC0)},
      {FORMAT_HWCN,
       std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByHWCN)},
      {FORMAT_ND,
       std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByND)},
      {FORMAT_C1HWNCoC0,
       std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByC1HWNCoC0)}};
}

int64_t DivisionCeiling(int64_t dividend, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else {
    return (dividend + divisor - 1) / divisor;
  }
}


bool AxisUtil::GetAxisValueByOriginFormat(const Format& format,
                                            const vector<int64_t>& dimVec,
                                            const uint32_t& c0,
                                            vector<int64_t>& axisValue,
                                            vector<int64_t>& ndValue) {
  auto iterGetAxisFunc = getAxisValueFuncMap.find(format);
  if (iterGetAxisFunc == getAxisValueFuncMap.end()) {
    LOG_INFO("Can not get axis value of old format %u!", format);
    return false;
  }
  GetAxisValueInfoByFormatPtr getAxisFunc = iterGetAxisFunc->second;
  CHECK_NOTNULL(getAxisFunc);
  return (*getAxisFunc)(dimVec, c0, axisValue, ndValue);
}

bool AxisUtil::HasAxisValueFunc(const Format& format){
    auto iterGetAxisFunc = getAxisValueFuncMap.find(format);
    if (iterGetAxisFunc == getAxisValueFuncMap.end()) {
        LOG_INFO("Can not get axis value of format %u!", format);
        return false;
    }
    return true;
}

bool AxisUtil::CheckParams(const vector<int64_t>& originalDimVec,
                     const uint32_t& c0,
                     vector<int64_t>& axisValue,
                     vector<int64_t>& ndValue) {
  ndValue = originalDimVec;
  auto dimSize = originalDimVec.size();
  if (dimSize < ge::DIM_DEFAULT_SIZE) {
    /* Before this funcion, we should call function PadDimensionTo4. */
    LOG_INFO("Dimension size %zu is invalid.", dimSize);
    return false;
  }
  if (c0 == 0) {
    LOG_ERROR("[ERROR]c0 is zero!");
    return false;
  }

  return true;
}

bool AxisUtil::GetAxisValueByND(const vector<int64_t>& originalDimVec,
                                  const uint32_t& c0,
                                  vector<int64_t>& axisValue,
                                  vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), LOG_INFO("Original dim vector is empty!"),
           return true);
  ndValue = originalDimVec;
  /* To differentiate the input datatype of int8 and others */
  axisValue[AXIS_C0] = c0;
  if (originalDimVec.size() == NCHW_DIMENSION_NUM) {
    axisValue[AXIS_N] = originalDimVec[AXIS_NCHW_DIM_N];
    axisValue[AXIS_C] = originalDimVec[AXIS_NCHW_DIM_C];
    axisValue[AXIS_H] = originalDimVec[AXIS_NCHW_DIM_H];
    axisValue[AXIS_W] = originalDimVec[AXIS_NCHW_DIM_W];
    axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NCHW_DIM_C],
                                         (int64_t)c0);
    axisValue[AXIS_Co] = c0;
  }
  return true;
}

bool AxisUtil::GetAxisValueByNCHW(const vector<int64_t>& originalDimVec,
                                    const uint32_t& c0,
                                    vector<int64_t>& axisValue,
                                    vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), LOG_INFO("Original dim vector is empty!"),
           return true);
  /* C0 Must be set for case ND or 2D-NCHW to NZ */
  axisValue[AXIS_C0] = c0;
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true,
          LOG_ERROR("[ERROR]Parameter is invalid!"), return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_NCHW_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_NCHW_DIM_C];
  axisValue[AXIS_H] = originalDimVec[AXIS_NCHW_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_NCHW_DIM_W];
  axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NCHW_DIM_C],
                                       (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByNHWC(const vector<int64_t>& originalDimVec,
                                    const uint32_t& c0,
                                    vector<int64_t>& axisValue,
                                    vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), LOG_INFO("Original dim vector is empty!"),
           return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true,
           LOG_ERROR("[ERROR]Parameter is invalid!"), return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_NHWC_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_NHWC_DIM_C];
  axisValue[AXIS_H] = originalDimVec[AXIS_NHWC_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_NHWC_DIM_W];
  axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NHWC_DIM_C],
                                       (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByNC1HWC0(const vector<int64_t>& originalDimVec,
                                       const uint32_t& c0,
                                       vector<int64_t>& axisValue,
                                       vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), LOG_INFO("Original dim vector is empty!"),
           return true);
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true,
           LOG_ERROR("[ERROR]Parameter is invalid!"), return false);

  auto dimSize = originalDimVec.size();
  if (dimSize == ge::DIM_DEFAULT_SIZE + 1) {
    axisValue[AXIS_C1] = originalDimVec[AXIS_NC1HWC0_DIM_C1];
    axisValue[AXIS_C0] = originalDimVec[AXIS_NC1HWC0_DIM_C0];
    axisValue[AXIS_C] = axisValue[AXIS_C1] * axisValue[AXIS_C0];
  } else {
    axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NCHW_DIM_C],
                                         (int64_t)c0);
    axisValue[AXIS_C0] = c0;
    axisValue[AXIS_C] = originalDimVec[AXIS_NCHW_DIM_C];
  }

  axisValue[AXIS_N] = originalDimVec[AXIS_NCHW_DIM_N];
  axisValue[AXIS_H] = originalDimVec[AXIS_NCHW_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_NCHW_DIM_W];
  return true;
}

bool AxisUtil::GetAxisValueByHWCN(const vector<int64_t>& originalDimVec,
                                  const uint32_t& c0,
                                  vector<int64_t>& axisValue,
                                  vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), LOG_INFO("Original dim vector is empty!"),
           return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true,
        LOG_ERROR("[ERROR]Parameter is invalid!"), return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_HWCN_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_HWCN_DIM_C];
  axisValue[AXIS_H] = originalDimVec[AXIS_HWCN_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_HWCN_DIM_W];
  axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_HWCN_DIM_C],
                                       (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByC1HWNCoC0(const vector<int64_t>& originalDimVec,
                                  const uint32_t& c0,
                                  vector<int64_t>& axisValue,
                                  vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), LOG_INFO("Original dim vector is empty!"),
           return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true,
        LOG_ERROR("[ERROR]Parameter is invalid!"), return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_C1HWNCoC0_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_C1HWNCoC0_DIM_C1] * c0;
  axisValue[AXIS_H] = originalDimVec[AXIS_C1HWNCoC0_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_C1HWNCoC0_DIM_W];
  axisValue[AXIS_C1] = originalDimVec[AXIS_C1HWNCoC0_DIM_C1];
  axisValue[AXIS_Co] = originalDimVec[AXIS_C1HWNCoC0_DIM_Co];
  return true;
}

}; // namespace ge
