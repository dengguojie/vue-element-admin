#include "quant_host_cpu_op_common.h"

namespace fe {
inline Status CheckInt64MulOverflowForPass(int64_t a, int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > ((int64_t)INT64_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < ((int64_t)INT64_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < ((int64_t)INT64_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < ((int64_t)INT64_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status GetkernelDataCountForPass(const std::vector<int64_t> &filterDIms,
                          int64_t &kernelDataCount) {
  for (size_t i = 0; i < filterDIms.size(); i++) {
    if(CheckInt64MulOverflowForPass(kernelDataCount, filterDIms.at(i)) != SUCCESS) {
        return FAILED;
    }
    kernelDataCount *= filterDIms.at(i);
  }
  return SUCCESS;
}
} // namespace fe