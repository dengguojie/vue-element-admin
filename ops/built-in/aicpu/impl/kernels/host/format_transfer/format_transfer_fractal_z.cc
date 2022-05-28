/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "format_transfer_fractal_z.h"

#include "format_transfer_utils.h"
#include "formats_definitions.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

using namespace std;

namespace aicpu {
namespace formats {
namespace {
bool CheckDataTypeSupport(DataType data_type) {
  return GetSizeByDataType(data_type) > 0 ? KERNEL_STATUS_OK
                                          : KERNEL_STATUS_PARAM_INVALID;
}

/**
 * FZ represents the weight of convolution,.
 * After the conversion to two-dimensional matrix, the memory arrangement is
 * small n and large Z. If 4D(eg.NCHW) is used to represent convolution kernel,
 * N is width, HWC is height.
 *
 * frac_z axises: (C1*H*W, No, Ni, C0), which Ni = 16, C0 = 16/32, No =
 * Ceil(N/Ni), C1 = Ceil(C/C0)
 * @return
 */

uint32_t TransShapeToFzWithGroups(int64_t n, int64_t c, int64_t h, int64_t w,
                                  DataType data_type,
                                  std::vector<int64_t> &dst_shape,
                                  int64_t groups) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    KERNEL_LOG_ERROR("Cube size must greater than or equal to 0");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t cin_ori = c;
  // For this place , groups is not equal to 0, which had been checked in [Transdata] entrance.
  int64_t cout_ori = n / groups;
  if (cin_ori == 0 || cout_ori == 0) {
    KERNEL_LOG_ERROR(
        "Cin_ori, cout_ori must not be equal 0, "
        "and current cin_ori, cout_ori, groups are [%ld] [%ld] [%ld]",
        cin_ori, cout_ori, groups);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t cube_k = GetCubeSizeByDataType(data_type);
  int64_t e_mult =
      std::min(Lcm(Lcm(cin_ori, cube_k) / (cin_ori),
                   Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / (cout_ori)),
               groups);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t c1_dim = cin_opt / cube_k;
  int64_t g_dim = Ceil(groups, e_mult);
  auto n1 = Ceil(cout_ori * e_mult, static_cast<int64_t>(kCubeSize));
  dst_shape.clear();
  dst_shape.push_back(g_dim * c1_dim * h * w);
  dst_shape.push_back(n1);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(cube_k);
  if (!IsShapeValid(dst_shape)) {
    KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]",
                     VectorToString(dst_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TransShapeNchwToFzWithGroups(const std::vector<int64_t> &src_shape,
                                      DataType data_type,
                                      std::vector<int64_t> &dst_shape,
                                      int64_t groups) {
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto n = src_shape.at(kNchwN);
  auto c = src_shape.at(kNchwC);
  auto h = src_shape.at(kNchwH);
  auto w = src_shape.at(kNchwW);
  return TransShapeToFzWithGroups(n, c, h, w, data_type, dst_shape, groups);
}

uint32_t TransShapeHwcnToFzWithGroups(const std::vector<int64_t> &src_shape,
                                      DataType data_type,
                                      std::vector<int64_t> &dst_shape,
                                      int64_t groups) {
  if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto h = src_shape.at(kHwcnH);
  auto w = src_shape.at(kHwcnW);
  auto c = src_shape.at(kHwcnC);
  auto n = src_shape.at(kHwcnN);

  return TransShapeToFzWithGroups(n, c, h, w, data_type, dst_shape, groups);
}

uint32_t TransShapeNhwcToFzWithGroups(const std::vector<int64_t> &src_shape,
                                      DataType data_type,
                                      std::vector<int64_t> &dst_shape,
                                      int64_t groups) {
  if (!CheckShapeValid(src_shape, kNhwcDimsNum)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto n = src_shape.at(kNhwcN);
  auto h = src_shape.at(kNhwcH);
  auto w = src_shape.at(kNhwcW);
  auto c = src_shape.at(kNhwcC);

  return TransShapeToFzWithGroups(n, c, h, w, data_type, dst_shape, groups);
}

// Supporting NHWC, NCHW, HWCN converte to FORMAT_FRACTAL_Z (GC1HWN1N0C0),
// the final effect achieved is for the data to be distributed diagonally.
// For example: When the input filter format is NCHW, calculated the Correspondence of
// index between NCHW and FORMAT_FRACTAL_Z , then Convert the old filter to the new
// filter, and finally added 0 to the position where there is no data.
uint32_t TransFormatToFzWithGroups(const TransArgs &args, TransResult &result,
                                   int64_t groups) {
  int64_t h_dim = 0;
  int64_t w_dim = 0;
  int64_t c_dim = 0;
  int64_t n_dim = 0;
  int64_t d_dim = 1;
  if (args.src_format == FORMAT_NHWC) {
    n_dim = args.src_shape.at(kNhwcN);
    h_dim = args.src_shape.at(kNhwcH);
    w_dim = args.src_shape.at(kNhwcW);
    c_dim = args.src_shape.at(kNhwcC);
  } else if (args.src_format == FORMAT_HWCN) {
    h_dim = args.src_shape.at(kHwcnH);
    w_dim = args.src_shape.at(kHwcnW);
    c_dim = args.src_shape.at(kHwcnC);
    n_dim = args.src_shape.at(kHwcnN);
  } else if (args.src_format == FORMAT_NCHW) {
    n_dim = args.src_shape.at(kNchwN);
    c_dim = args.src_shape.at(kNchwC);
    h_dim = args.src_shape.at(kNchwH);
    w_dim = args.src_shape.at(kNchwW);
  } else {
    KERNEL_LOG_WARN(
        "Supporting format conversion from NHWC, NCHW and HWCN to FRACTAL_Z, current input "
        "format is [%s]",
        FormatToSerialString(args.src_format).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t cin_ori = c_dim;
  // For this place , groups is not equal to 0, which had been checked in [Transdata] entrance.
  int64_t cout_ori = n_dim / groups;
  if (cin_ori == 0 || cout_ori == 0) {
    KERNEL_LOG_ERROR(
        "Cin_ori, cout_ori must not be equal 0, "
        "and current cin_ori, cout_ori, groups are [%ld] [%ld] [%ld]",
        cin_ori, cout_ori, groups);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const int64_t cube_k = GetCubeSizeByDataType(args.src_data_type);
  int64_t e_mult =
      std::min(Lcm(Lcm(cin_ori, cube_k) / (cin_ori),
                   Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / (cout_ori)),
               groups);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t cout_opt = Ceil(e_mult * cout_ori, static_cast<int64_t>(kCubeSize)) *
                     static_cast<int64_t>(kCubeSize);
  int64_t c1_dim = cin_opt / cube_k;
  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = GetItemNumByShape(args.dst_shape) * data_size;
  // The input is empty tensor, we should return sucess directly.
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return KERNEL_STATUS_OK;
  }
  errno_t ret = EOK;
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size],
                               std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    KERNEL_LOG_ERROR(
        "Failed to allcoate memory for dst buf [%ld] when trans format from [%s] to [%s]",
        dst_size, FormatToSerialString(args.src_format).c_str(),
        FormatToSerialString(args.dst_format).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  ret = memset_s(dst.get(), static_cast<size_t>(dst_size), 0,
                 static_cast<size_t>(dst_size));
  if (ret != EOK) {
    KERNEL_LOG_ERROR("Failed to assign 0, ret is [%d]", ret);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (int64_t g = 0; g < groups; g++) {
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
              int64_t dst_inx =
                  (g / e_mult) * d_dim * c1_dim * h_dim * w_dim * cout_opt *
                      cube_k +
                  d * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                  (dst_ci / cube_k) * h_dim * w_dim * cout_opt * cube_k +
                  h * w_dim * cout_opt * cube_k + w * cout_opt * cube_k +
                  dst_co * cube_k + tempory;
              if (args.src_format == FORMAT_HWCN) {
                srx_inx = d * h_dim * w_dim * c_dim * n_dim +
                          h * w_dim * c_dim * n_dim + w * c_dim * n_dim +
                          c * n_dim + src_co;
              } else if (args.src_format == FORMAT_NCHW) {
                srx_inx = src_co * c_dim * d_dim * h_dim * w_dim +
                          c * d_dim * h_dim * w_dim + d * h_dim * w_dim +
                          h * w_dim + w;
              } else if (args.src_format == FORMAT_NHWC) {
                srx_inx = src_co * d_dim * h_dim * w_dim * c_dim +
                          d * h_dim * w_dim * c_dim + h * w_dim * c_dim +
                          w * c_dim + c;
              }
              char *dst_data =
                  reinterpret_cast<char *>(dst.get() + dst_inx * data_size);
              const char *src_data = reinterpret_cast<const char *>(
                  args.data + srx_inx * data_size);
              for (int64_t index = 0; index < data_size; index++) {
                *dst_data++ = *src_data++;
              }
            }
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return KERNEL_STATUS_OK;
}
}  // namespace

uint32_t FormatTransferFractalZ::TransFormat(const TransArgs &args,
                                             TransResult &result) {
  KERNEL_LOG_INFO(
      "Begin to trans format from [%s] to [%s], src shape [%s], data type [%s], dst "
      "shape [%s]",
      FormatToSerialString(args.src_format).c_str(),
      FormatToSerialString(args.dst_format).c_str(),
      VectorToString(args.src_shape).c_str(),
      DTypeStr(args.src_data_type).c_str(),
      VectorToString(args.dst_shape).c_str());

  if (args.groups == 0) {
    KERNEL_LOG_ERROR("Attr[groups] must not be equal to 0");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  std::vector<int64_t> expect_shape;
  auto ret = TransShape(args.src_format, args.src_shape, args.src_data_type,
                        args.dst_format, expect_shape, args.groups);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  if (!IsTransShapeDstCorrect(args, expect_shape)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (((args.src_format == FORMAT_NHWC) || (args.src_format == FORMAT_HWCN) ||
       (args.src_format == FORMAT_NCHW)) &&
      args.dst_format == FORMAT_FRACTAL_Z) {
    return TransFormatToFzWithGroups(args, result, args.groups);
  }
  return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t FormatTransferFractalZ::TransShape(
    Format src_format, const std::vector<int64_t> &src_shape,
    DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape,
    int64_t groups) {
  if (CheckDataTypeSupport(data_type) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (src_format == FORMAT_NHWC &&
      GetPrimaryFormat(dst_format) == FORMAT_FRACTAL_Z) {
    return TransShapeNhwcToFzWithGroups(src_shape, data_type, dst_shape,
                                        groups);
  }
  if ((src_format == FORMAT_HWCN) &&
      (GetPrimaryFormat(dst_format) == FORMAT_FRACTAL_Z)) {
    return TransShapeHwcnToFzWithGroups(src_shape, data_type, dst_shape,
                                        groups);
  }
  if (src_format == FORMAT_NCHW &&
      GetPrimaryFormat(dst_format) == FORMAT_FRACTAL_Z) {
    return TransShapeNchwToFzWithGroups(src_shape, data_type, dst_shape,
                                        groups);
  }

  return KERNEL_STATUS_PARAM_INVALID;
}
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NCHW, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_HWCN, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NHWC, FORMAT_FRACTAL_Z)
}  // namespace formats
}  // namespace  aicpu