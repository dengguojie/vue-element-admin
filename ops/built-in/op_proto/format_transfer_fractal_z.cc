/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "format_transfer_fractal_z.h"
#include "op_log.h"
#include <securec.h>
#include <memory>
#include "external/graph/types.h"
#include "graph/utils/type_utils.h"
#include "ge/ge_api_error_codes.h"
namespace {

static const int kCubeSize = 16;
static const int kNiSize = 16;

enum NdDimIndex { k2dC, k2dN, k2dDimsNum };
static const int64_t kShapeItemNumMAX = 1024UL * 1024UL * 1024UL * 1024UL;

template <typename T>
T Ceil(T n1, T n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? (n1 - 1) / n2 + 1 : 0;
}

int64_t GetCubeSizeByDataType(ge::DataType data_type) {
  // Current cube does not support 4 bytes and longer data
  auto size = GetSizeByDataType(data_type);
  if (size <= 0) {
    return -1;
  } else if (size == 1) {
    return kCubeSize * 2;  // 32 bytes cube size
  } else {
    return kCubeSize;
  }
}

bool IsShapeValid(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return false;
  }
  int64_t num = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      return false;
    }
    if (dim != 0 && kShapeItemNumMAX / dim < num) {
      return false;
    }
    num *= dim;
  }
  return true;
}

bool CheckShapeValid(const std::vector<int64_t> &shape, const int64_t expect_dims) {
  if (expect_dims <= 0 || shape.size() != static_cast<size_t>(expect_dims)) {
    return false;
  }
  return IsShapeValid(shape);
}
}

namespace ge {
namespace formats {
namespace {
Status CheckDataTypeSupport(DataType data_type) { return GetSizeByDataType(data_type) > 0 ? SUCCESS : FAILED; }

/**
 * FZ represents the weight of convolution,.
 * After the conversion to two-dimensional matrix, the memory arrangement is small n and large Z.
 * If 4D(eg.NCHW) is used to represent convolution kernel, N is width, HWC is height.
 *
 * frac_z axises: (C1*H*W, No, Ni, C0), which Ni = 16, C0 = 16/32, No = Ceil(N/Ni), C1 = Ceil(C/C0)
 * @return
 */
Status TransShapeToFz(int64_t n, int64_t c, int64_t h, int64_t w, DataType data_type, std::vector<int64_t> &dst_shape) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    OP_LOGE("data_type is valid");
    return FAILED;
  }

  auto c1 = Ceil(c, c0);
  auto no = Ceil(n, static_cast<int64_t>(kNiSize));

  dst_shape.clear();
  dst_shape.push_back(h * w * c1);
  dst_shape.push_back(no);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(c0);
  if (!IsShapeValid(dst_shape)) {
    OP_LOGE("dst_shape is valid");
    return FAILED;
  }
  return SUCCESS;
}

Status TransShapeNdToFz(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, k2dDimsNum)) {
    OP_LOGE("src_shape is valid");
    return FAILED;  // Only support 2D to fracz
  }
  auto h = 1;
  auto w = 1;  // The shape conversion in 2d format is the same as 1,1,c,n
  auto c = src_shape.at(k2dC);
  auto n = src_shape.at(k2dN);

  return TransShapeToFz(n, c, h, w, data_type, dst_shape);
}

Status TransFormatNdToFz(const TransArgs &args, TransResult &result) {
  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = 1;
  for (auto dim : args.dst_shape) {
    dst_size *= dim;
  }
  dst_size *= data_size;
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return SUCCESS;
  }

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    OP_LOGE("dst == nullptr");
    return FAILED;
  }

  int64_t c = args.src_shape[k2dC];
  int64_t n = args.src_shape[k2dN];
  int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);
  auto hwn1n0c0 = n1n0 * c0;
  for (int64_t c1i = 0; c1i < c1; c1i++) {
    for (int64_t n1n0i = 0; n1n0i < n1n0; n1n0i++) {
      for (int64_t c0i = 0; c0i < c0; c0i++) {
        int64_t dst_idx = c1i * hwn1n0c0 + n1n0i * c0 + c0i;
        int64_t dst_offset = dst_idx * data_size;
        auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                  ? dst_size - dst_offset
                                  : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
        auto pad_zero = ((c1i * c0 + c0i) >= c) || (n1n0i >= n);
        errno_t ret;
        if (pad_zero) {
          ret =
              memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0, static_cast<size_t>(data_size));
        } else {
          int64_t src_idx = (c1i * c0 + c0i) * n + n1n0i;
          ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_idx * data_size,
                         static_cast<size_t>(data_size));
        }
        if (ret != EOK) {
          OP_LOGE("ret != EOK");
          return FAILED;
        }
      }
    }
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}

}  // namespace

Status FormatTransferFractalZ_TBE::TransFormat(const TransArgs &args, TransResult &result) {
  std::vector<int64_t> expect_shape;
  auto ret = TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, expect_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!args.dst_shape.empty() && args.dst_shape != expect_shape) {
    OP_LOGE("dst_shape id empty or valid");
    return FAILED;
  }

  if (args.src_format == FORMAT_ND && args.dst_format == FORMAT_FRACTAL_Z) {
    return TransFormatNdToFz(args, result);
  }

  return FAILED;
}

Status FormatTransferFractalZ_TBE::TransShape(Format src_format, const std::vector<int64_t> &src_shape, DataType data_type,
                                          Format dst_format, std::vector<int64_t> &dst_shape) {
  if (CheckDataTypeSupport(data_type) != SUCCESS) {
    return FAILED;
  }

  if (src_format == FORMAT_ND && dst_format == FORMAT_FRACTAL_Z) {
    return TransShapeNdToFz(src_shape, data_type, dst_shape);
  }

  return FAILED;
}
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ_TBE, FORMAT_ND, FORMAT_FRACTAL_Z)

}  // namespace formats
}  // namespace ge
