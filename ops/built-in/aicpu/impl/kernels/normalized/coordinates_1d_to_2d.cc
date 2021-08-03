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
#include "coordinates_1d_to_2d.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kCoordinates1DTo2D = "Coordinates1DTo2D";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 3;

template <typename T>
void Coordinates1DTo2D(const T *x_data, const T *shape_data, T *row_data,
                       T *col_data, T *n_data) {
  T x = *x_data;
  T col_num = shape_data[3];
  *row_data = x / col_num;
  *col_data = x % col_num;
  *n_data = col_num;
  KERNEL_LOG_INFO(
      "Input x[%lld], shape row[%lld], shape col[%lld], "
      "output row index[%lld], output col index[%lld], output n[%lld].",
      static_cast<int64_t>(x), static_cast<int64_t>(shape_data[2]),
      static_cast<int64_t>(shape_data[3]), static_cast<int64_t>(*row_data),
      static_cast<int64_t>(*col_data), *n_data);
}
}  // namespace

namespace aicpu {
uint32_t Coordinates1DTo2DCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Coordinates1DTo2D params failed.");
  Tensor *x = ctx.Input(0);
  Tensor *shape = ctx.Input(1);
  Tensor *output_row = ctx.Output(0);
  Tensor *output_col = ctx.Output(1);
  Tensor *output_n = ctx.Output(2);
  DataType x_dt = x->GetDataType();
  DataType shape_dt = shape->GetDataType();
  KERNEL_CHECK_FALSE((x_dt == shape_dt), KERNEL_STATUS_INNER_ERROR,
                     "Input[x] data type[%s] and input[shape] data type[%s] "
                     "must be same.",
                     DTypeStr(x_dt).c_str(), DTypeStr(shape_dt).c_str());

  KERNEL_CHECK_FALSE((shape->NumElements() == 4), KERNEL_STATUS_INNER_ERROR,
                     "Input[shape] element number must be equal to 4, "
                     "but got[%lld].",
                     shape->NumElements());
  uint32_t ret = KERNEL_STATUS_OK;
  switch (x_dt) {
    case DT_INT32:
      Coordinates1DTo2D(static_cast<int32_t *>(x->GetData()),
                        static_cast<int32_t *>(shape->GetData()),
                        static_cast<int32_t *>(output_row->GetData()),
                        static_cast<int32_t *>(output_col->GetData()),
                        static_cast<int32_t *>(output_n->GetData()));
      break;
    case DT_INT64:
      Coordinates1DTo2D(static_cast<int64_t *>(x->GetData()),
                        static_cast<int64_t *>(shape->GetData()),
                        static_cast<int64_t *>(output_row->GetData()),
                        static_cast<int64_t *>(output_col->GetData()),
                        static_cast<int64_t *>(output_n->GetData()));
      break;
    case DT_UINT64:
      Coordinates1DTo2D(static_cast<uint64_t *>(x->GetData()),
                        static_cast<uint64_t *>(shape->GetData()),
                        static_cast<uint64_t *>(output_row->GetData()),
                        static_cast<uint64_t *>(output_col->GetData()),
                        static_cast<uint64_t *>(output_n->GetData()));
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[x] data type[%s]",
                       DTypeStr(x_dt).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kCoordinates1DTo2D, Coordinates1DTo2DCpuKernel);
}  // namespace aicpu
