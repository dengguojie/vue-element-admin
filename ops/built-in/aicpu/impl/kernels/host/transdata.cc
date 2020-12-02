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
#include "transdata.h"

#include <algorithm>

#include "utils/kernel_util.h"
#include "Eigen/Core"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace std;

namespace {
const char *TRANS_DATA = "TransData";
constexpr int64_t kDimN0 = 16;
constexpr int64_t kCubeN = 16;
constexpr int64_t kGroupNum = 1;

inline int64_t Measure(int64_t x, int64_t y) {
  int64_t z = y;
  while (x % y != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}

// least common multiple 
inline int64_t Lcm(int64_t a, int64_t b) {
  if (b == 0) {
    return -1;
  }
  int64_t temp = (a * b) / (Measure(a, b));
  return temp;
}

// get the result of two number divisor and let result round up
inline int64_t Ceil(int64_t a, int64_t b) {
  if (b == 0) {
    return -1;
  } else {
    int64_t ret = a / b;
    if ((a % b) != 0) {
      ret++;
    }
    return ret;
  }
}
}  // namespace

namespace aicpu {
template <typename T>
uint32_t TransDataCpuKernel::DealData(T *input_data, T *output_data, 
      Tensor *input_tensor, Tensor *output_tensor, int64_t group) {
  DataType dt = static_cast<DataType>(input_tensor->GetDataType());
  // if cube_k equals to DT_INT8, and let its values 32 else if equals to
  // DT_FLOAT16 or DT_FLOAT ,need to let its values 16. other dateType not 
  // support compute
  KERNEL_CHECK_FALSE(
      ((dt == DT_FLOAT16) || (dt == DT_INT8) || (dt == DT_FLOAT)),
      KERNEL_STATUS_PARAM_INVALID,
      "Input type is not DT_INT8 or DT_FLOAT16 or DT_FLOAT [%d]", dt);
  const int64_t cube_k = dt == DT_INT8  ? 32 : 16;
  auto input_format = input_tensor->GetTensorShape()->GetFormat();
  std::vector<int64_t> dims;
  dims = input_shape->GetDimSizes();
  KERNEL_CHECK_FALSE((dims.size() >= 4), KERNEL_STATUS_PARAM_INVALID,
      "%s Dims size [%zu] must >= 4", TRANS_DATA, dims.size());
  int64_t d_dim;
  int64_t h_dim;
  int64_t w_dim;
  int64_t c_dim;
  int64_t n_dim;
  if (input_format == FORMAT_NCDHW) {
    n_dim = dims[0];
    c_dim = dims[1];
    d_dim = dims[2];
    h_dim = dims[3];
    w_dim = dims[4];
  } else if (input_format == FORMAT_DHWCN) {
    d_dim = dims[0];
    h_dim = dims[1];
    w_dim = dims[2];
    c_dim = dims[3];
    n_dim = dims[4];
  } else if (input_format == FORMAT_NDHWC) {
    n_dim = dims[0];
    d_dim = dims[1];
    h_dim = dims[2];
    w_dim = dims[3];
    c_dim = dims[4];
  } else if (input_format == FORMAT_NHWC) {
    n_dim = dims[0];
    h_dim = dims[1];
    d_dim = 1;
    w_dim = dims[2];
    c_dim = dims[3];
  } else if (input_format == FORMAT_NCHW) {
    n_dim = dims[0];
    c_dim = dims[1];
    h_dim = dims[2];
    w_dim = dims[3];
    d_dim = 1;
  } else if (input_format == FORMAT_HWCN) {
    h_dim = dims[0];
    w_dim = dims[1];
    c_dim = dims[2];
    n_dim = dims[3];
    d_dim = 1;
  } 
  else {
    KERNEL_LOG_ERROR(
        "Format is not FORMAT_DHWCN or FORMAT_NDHWC or FORMAT_NCDHW or "
        "FORMAT_NHWC or FORMAT_NCHW, current input format is [%d]",
        input_format);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int64_t cin_ori = c_dim;
  int64_t cout_ori = n_dim / group;
  if (cin_ori == 0 || cout_ori == 0) {
    KERNEL_LOG_ERROR("Cin_ori or cout_ori must be not equal 0");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int64_t e_mult = std::min(
      Lcm(Lcm(cin_ori, cube_k) / (cin_ori), Lcm(cout_ori, kCubeN) / (cout_ori)),
      group);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t cout_opt = Ceil(e_mult * cout_ori, kCubeN) * kCubeN;
  int64_t c1_dim = cin_opt / cube_k;
  int64_t g_dim = Ceil(group, e_mult);
  int64_t dim_cin = cin_opt / cube_k;
  int64_t size_output_data =
      g_dim * d_dim * dim_cin * h_dim * w_dim * cout_opt * cube_k;
  memset_s(output_data, sizeof(T) * size_output_data, 0,
           sizeof(T) * size_output_data);
  for (int64_t g = 0; g < group; g++) {
    for (int64_t d = 0; d < d_dim; d++) {
      for (int64_t c = 0; c < c_dim; c++) {
        for (int64_t h = 0; h < h_dim; h++) {
          for (int64_t w = 0; w < w_dim; w++) {
            for (int64_t n = 0; n < cout_ori; n++) {
              int64_t e_val = g % e_mult;
              int64_t dst_ci = e_val * cin_ori + c;
              int64_t dst_co = e_val * cout_ori + n;
              int64_t src_co = g * cout_ori + n;
              int64_t tempory = dst_ci % cube_k;
              int64_t srx_inx = 0;
              int64_t dst_inx = (g / e_mult) * d_dim * c1_dim * h_dim * w_dim * 
                                cout_opt * cube_k + d * c1_dim * h_dim * w_dim *
                                cout_opt * cube_k + (dst_ci / cube_k) * h_dim *
                                w_dim * cout_opt * cube_k + h * w_dim * cout_opt
                                * cube_k + w * cout_opt * cube_k + dst_co *
                                * cube_k + tempory;
              if ((input_format == FORMAT_DHWCN) ||
                    (input_format == FORMAT_HWCN)) {
                srx_inx = d * h_dim * w_dim * c_dim * n_dim + h * w_dim * c_dim 
                          * n_dim + w * c_dim * n_dim + c * n_dim + src_co;
              } else if ((input_format == FORMAT_NCDHW) ||
                            (input_format == FORMAT_NCHW)) {
                srx_inx = src_co * c_dim * d_dim * h_dim * w_dim + c * d_dim * 
                          h_dim * w_dim + d * h_dim * w_dim + h * w_dim + w;
              } else if ((input_format == FORMAT_NDHWC) ||
                            (input_format == FORMAT_NHWC)) {
                srx_inx = src_co * d_dim * h_dim * w_dim * c_dim + d * h_dim * 
                          w_dim * c_dim + h * w_dim * c_dim + w * c_dim + c;
              }
              output_data[dst_inx] = input_data[srx_inx];
            }
          }
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::Compute(CpuKernelContext& ctx) {
  Tensor* input_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "%s Get Tensor:input_tensor failed", TRANS_DATA);
  Tensor* output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "%s Get Tensor:output_tensor failed", TRANS_DATA);
  auto output_format = output_tensor->GetTensorShape()->GetFormat();
  if((output_format != FORMAT_FRACTAL_Z) && (output_format != FORMAT_FRACTAL_Z_3D)) {
    KERNEL_LOG_EVENT("%s Unsupport output_format [%d]", 
                    TRANS_DATA , output_format);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  AttrValue* groups = ctx.GetAttr("groups");
  int64_t group = kGroupNum;
  if (groups != nullptr) {
    group = groups->GetInt();
  }
  DataType dt = static_cast<DataType>(input_tensor->GetDataType());
  auto input_data_temp = input_tensor->GetData();
  KERNEL_CHECK_NULLPTR(input_data_temp, KERNEL_STATUS_PARAM_INVALID,
                       "%s Get Tensor:input_data_temp failed", TRANS_DATA);
  auto output_data_temp = output_tensor->GetData();
  KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_PARAM_INVALID,
                       "%s Get Tensor:input_data_temp failed", TRANS_DATA);
  switch (dt) {
    case DT_INT8:
      return DealData(reinterpret_cast<int8_t*>(input_data_temp), 
                      reinterpret_cast<int8_t*>(output_data_temp),
                      input_tensor, output_tensor, group);
      break;
    case DT_FLOAT:
      return DealData(reinterpret_cast<float*>(input_data_temp), 
                      reinterpret_cast<float*>(output_data_temp),
                      input_tensor, output_tensor, group);
      break;
    case DT_FLOAT16:
      return DealData(reinterpret_cast<Eigen::half*>(input_data_temp), 
                      reinterpret_cast<Eigen::half*>(output_data_temp),
                      input_tensor, output_tensor, group);
      break;

    default:
      KERNEL_LOG_ERROR(
          "DateType is not DT_INT8 or DT_FLOAT or DT_FLOAT16, and current "
          "DataType is [%d]",
          dt);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
REGISTER_CPU_KERNEL(TRANS_DATA, TransDataCpuKernel);
}  // namespace aicpu