
#include "cpu_kernel.h"
#include "utils/bcast.h"

#ifndef AICPU_UTILS_EQUAL_UTIL_H_
#define AICPU_UTILS_EQUAL_UTIL_H_
namespace aicpu {
   /**
   * @brief Parameter verification
   * @param flag equal or not equal
   * @return status code
   */
template <typename T>
uint32_t EqualCalculate(CpuKernelContext &ctx,
                                      BCalcInfo &calc_info, bool flag) {
  auto input_x1 = reinterpret_cast<T *>(calc_info.input_0->GetData());
  auto input_x2 = reinterpret_cast<T *>(calc_info.input_1->GetData());
  auto output_y = reinterpret_cast<bool *>(calc_info.output->GetData());
  KERNEL_CHECK_NULLPTR(input_x1, KERNEL_STATUS_PARAM_INVALID,
                           "Get input x1 data failed.")
  KERNEL_CHECK_NULLPTR(input_x2, KERNEL_STATUS_PARAM_INVALID,
                           "Get input x2 data failed.")
  KERNEL_CHECK_NULLPTR(output_y, KERNEL_STATUS_PARAM_INVALID,
                           "Get output data failed.")
  size_t data_num = calc_info.x_indexes.size();
  auto shard_equal = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto x_index = input_x1 + calc_info.x_indexes[i];
      auto y_index = input_x2 + calc_info.y_indexes[i];
      output_y[i] = (flag == true) ? (*x_index == *y_index) : (*x_index != *y_index);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, 1, shard_equal),
                      "Equal calculate failed.")
  return KERNEL_STATUS_OK;
}
   /**
   * @brief Parameter verification
   * @param ctx op context
   * @param flag equal or not equal
   * @return status code
   */
template <typename T>
uint32_t EqualCompute(CpuKernelContext &ctx, bool flag) {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(0);
  calc_info.input_1 = ctx.Input(1);
  calc_info.output = ctx.Output(0);
  DataType input0_type = calc_info.input_0->GetDataType();
  DataType input1_type = calc_info.input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "DataType of x1 [%d] should be same as x2 [%d].",
                     input0_type, input1_type)
  KERNEL_LOG_INFO(
      "CpuKernel[%s], input x1 : addr[%p], size[%llu];"
      "input x2: addr[%p], size[%llu];"
      "output: addr[%p], size[%llu].",
      ctx.GetOpType().c_str(), calc_info.input_0->GetData(),
      calc_info.input_0->GetDataSize(), calc_info.input_1->GetData(),
      calc_info.input_1->GetDataSize(), calc_info.output->GetData(),
      calc_info.output->GetDataSize());

  Bcast bcast;
  KERNEL_HANDLE_ERROR(bcast.GenerateBcastInfo(calc_info),
                      "Generate broadcast info failed.")
  (void)bcast.BCastIndexes(calc_info.x_indexes, calc_info.y_indexes);
  (void)bcast.GetBcastVec(calc_info);

  return EqualCalculate<T>(ctx, calc_info, flag);
}
}
#endif
