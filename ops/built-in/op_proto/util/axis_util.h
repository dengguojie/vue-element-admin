/**
 * @file axis_util.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief get the axis value
 *
 * @version 1.0
 *
 */

#ifndef AXIS_UTIL_H
#define AXIS_UTIL_H

#include <functional>
#include <vector>
#include <memory.h>
#include "framework/omg/omg_inner_types.h"
#include "operator.h"
#include "op_log.h"
#include "graph/operator_reg.h"

#define LOG_ERROR(format, args...) printf(format, ##args)
#define LOG_INFO(format, args...) printf(format, ##args)
namespace ge {
const uint32_t NCHW_DIMENSION_NUM = 4;

const int32_t AXIS_NCHW_DIM_N = 0;
const int32_t AXIS_NCHW_DIM_C = 1;
const int32_t AXIS_NCHW_DIM_H = 2;
const int32_t AXIS_NCHW_DIM_W = 3;

const int32_t AXIS_NHWC_DIM_N = 0;
const int32_t AXIS_NHWC_DIM_H = 1;
const int32_t AXIS_NHWC_DIM_W = 2;
const int32_t AXIS_NHWC_DIM_C = 3;

const int32_t AXIS_NC1HWC0_DIM_N = 0;
const int32_t AXIS_NC1HWC0_DIM_C1 = 1;
const int32_t AXIS_NC1HWC0_DIM_C0 = 4;
const int32_t AXIS_NC1HWC0_DIM_H = 2;
const int32_t AXIS_NC1HWC0_DIM_W = 3;

const int32_t AXIS_HWCN_DIM_H = 0;
const int32_t AXIS_HWCN_DIM_W = 1;
const int32_t AXIS_HWCN_DIM_C = 2;
const int32_t AXIS_HWCN_DIM_N = 3;

const int32_t AXIS_C1HWNCoC0_DIM_C1 = 0;
const int32_t AXIS_C1HWNCoC0_DIM_H = 1;
const int32_t AXIS_C1HWNCoC0_DIM_W = 2;
const int32_t AXIS_C1HWNCoC0_DIM_N = 3;
const int32_t AXIS_C1HWNCoC0_DIM_Co = 4;
const int32_t AXIS_C1HWNCoC0_DIM_C0 = 5;

#define CHECK_NOTNULL(val)                                  \
  do { if ((val) == nullptr) {                                 \
  LOG_ERROR("[ERROR]Parameter[%s] must not be null.", #val);            \
  return false;   } } while (0)

#define CHECK(cond, log_func, return_expr) \
  do {\
    if (cond) {\
      log_func;\
      return_expr;\
    }\
  } while (0)

enum AxisValueType {
  AXIS_N = 0,
  AXIS_C = 1,
  AXIS_H = 2,
  AXIS_W = 3,
  AXIS_C1 = 4,
  AXIS_C0 = 5,
  AXIS_Co = 6,
  AXIS_D = 7,
  AXIS_BOTTOM = 8
};

int64_t DivisionCeiling(int64_t dividend, int64_t divisor);

/* Axis value is arranged as {N,C,H,W,C1,C0,...} */
/* The first parameter is old shape's dimension,
 * second is c0 and third is axis value. */
using GetAxisValueInfoByFormat =
    std::function<bool(const std::vector<int64_t>&, const uint32_t&,
                         std::vector<int64_t>&, std::vector<int64_t>&)>;

using GetAxisValueInfoByFormatPtr = std::shared_ptr<GetAxisValueInfoByFormat>;

class AxisUtil {
  public:
  AxisUtil();
  ~AxisUtil() {};
  bool GetAxisValueByOriginFormat(
      const ge::Format& format, const std::vector<int64_t>& dimVec,
      const uint32_t& c0, std::vector<int64_t>& axisValue, std::vector<int64_t>& ndValue);
  bool HasAxisValueFunc(const ge::Format& format);
  private:
  static bool CheckParams(const std::vector<int64_t>& originalDimVec,
                                   const uint32_t& c0,
                                   std::vector<int64_t>& axisValue,
                                   std::vector<int64_t>& ndValue);

  static bool GetAxisValueByNCHW(const std::vector<int64_t>& originalDimVec,
                                   const uint32_t& c0,
                                   std::vector<int64_t>& axisValue,
                                   std::vector<int64_t>& ndValue);

  static bool GetAxisValueByNHWC(const std::vector<int64_t>& originalDimVec,
                                   const uint32_t& c0,
                                   std::vector<int64_t>& axisValue,
                                   std::vector<int64_t>& ndValue);

  static bool GetAxisValueByNC1HWC0(const std::vector<int64_t>& originalDimVec,
                                      const uint32_t& c0,
                                      std::vector<int64_t>& axisValue,
                                      std::vector<int64_t>& ndValue);

  static bool GetAxisValueByFz(const std::vector<int64_t>& originalDimVec,
                                 const uint32_t& c0, std::vector<int64_t>& axisValue,
                                 std::vector<int64_t>& ndValue);

  static bool GetAxisValueByHWCN(const std::vector<int64_t>& originalDimVec,
                                   const uint32_t& c0,
                                   std::vector<int64_t>& axisValue,
                                   std::vector<int64_t>& ndValue);

  static bool GetAxisValueByND(const std::vector<int64_t>& originalDimVec,
                                   const uint32_t& c0,
                                   std::vector<int64_t>& axisValue,
                                   std::vector<int64_t>& ndValue);

  static bool GetAxisValueByC1HWNCoC0(const std::vector<int64_t>& originalDimVec,
                                   const uint32_t& c0,
                                   std::vector<int64_t>& axisValue,
                                   std::vector<int64_t>& ndValue);

  /* map of GetAxisValueInfoByFormat, get axis value by different original
   * formats. */
  std::map<ge::Format, GetAxisValueInfoByFormatPtr>
      getAxisValueFuncMap;
};
}  // namespace ge

#endif  // AXIS_UTIL_H

