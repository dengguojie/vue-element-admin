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
#include "transdata.h"

#include <algorithm>

#include "Eigen/Core"
#include "cpu_types.h"
#include "format_transfer/format_transfer_utils.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char *kTransData = "TransData";
constexpr int64_t kDimN0 = 16;
constexpr int64_t kCubeN = 16;
constexpr int64_t kGroupNum = 1;
constexpr int64_t kMaxDimsNumC = 4;
constexpr int32_t kCubeSize = 16;
constexpr int32_t kShapeOffset = 2;

template <typename T>
std::string VectorToString(const std::vector<T> &vec) {
  std::stringstream ss;
  bool first = true;
  for (auto &ele : vec) {
    if (first) {
      first = false;
    } else {
      ss << ",";
    }
    ss << ele;
  }
  return ss.str();
}

int64_t VectorToNum(const std::vector<int64_t> &vec) {
  int64_t result = 1;
  for (auto &ele : vec) {
    result *= ele;
  }
  return result;
}

void TransShapeByPerm(const std::vector<int64_t> &src_shape,
                      const std::vector<int64_t> &perm_arg,
                      std::vector<int64_t> &dst_shape) {
  dst_shape.resize(src_shape.size());
  for (size_t i = 0; i < perm_arg.size(); ++i) {
    dst_shape[i] = src_shape[perm_arg[i]];
  }
}

void GetIndexMap(const std::vector<int64_t> &perm_arg,
                 std::map<int32_t, int32_t> &index_map) {
  for (size_t i = 0; i < perm_arg.size(); i++) {
    index_map[perm_arg[i]] = i;
  }
}

void GetShapeHead(const std::vector<int64_t> &shape,
                  std::vector<int64_t> &shape_head) {
  shape_head.resize(shape.size());
  shape_head[shape.size() - 1] = 1;
  for (int i = shape.size() - kShapeOffset; i >= 0; i--) {
    shape_head[i] = shape_head[i + 1] * shape[i + 1];
  }
}

int32_t GetSrcIndex(int32_t dst_index, const std::vector<int64_t> &src_shape,
                    const std::vector<int64_t> &dst_shape,
                    const std::vector<int64_t> &src_shape_head,
                    const std::vector<int64_t> &dst_shape_head,
                    std::map<int32_t, int32_t> index_map) {
  std::vector<int32_t> src_vec(dst_shape.size());
  for (size_t i = 0; i < dst_shape.size(); i++) {
    src_vec[i] = dst_index / dst_shape_head[i];
    dst_index = dst_index % dst_shape_head[i];
  }
  int32_t src_index = 0;
  for (size_t i = 0; i < src_shape.size(); i++) {
    src_index += src_shape_head[i] * src_vec[index_map[i]];
  }
  return src_index;
}

static int64_t Measure(int64_t x, int64_t y) {
  int64_t z = y;
  while (x % y != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}

// least common multiple
static int64_t Lcm(int64_t a, int64_t b) {
  if (b == 0) {
    return -1;
  }
  int64_t temp = (a * b) / (Measure(a, b));
  return temp;
}

// get the result of two number divisor and let result round up
static int64_t Ceil(int64_t a, int64_t b) {
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
bool TransDataCpuKernel::IsOriginSupportFormatTransfer(Format src_format,
                                                       Format dst_format) {
  static const map<Format, map<Format, int32_t>> kOriginSupportFormatTransfer =
      {{FORMAT_HWCN, {{FORMAT_FRACTAL_Z_C04, 1}}}};
  auto dst = kOriginSupportFormatTransfer.find(src_format);
  if (dst == kOriginSupportFormatTransfer.end()) {
    return false;
  }
  return dst->second.count(dst_format) > 0;
}

uint32_t TransDataCpuKernel::NewCompute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "%s get input_tensor failed, input_tensor is nullptr.",
                       kTransData);
  uint8_t *input_data = reinterpret_cast<uint8_t *>(input_tensor->GetData());
  auto input_data_type = input_tensor->GetDataType();
  auto input_shape = input_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID,
                       "%s get input_shape failed, input_shape is nullptr.",
                       kTransData);
  auto input_dims = input_shape->GetDimSizes();
  auto input_format = input_shape->GetFormat();

  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "%s get output_tensor failed, output_tensor is nullptr.",
                       kTransData);
  auto output_data_type = output_tensor->GetDataType();
  auto output_shape = output_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(output_shape, KERNEL_STATUS_PARAM_INVALID,
                       "%s get output_shape failed, output_shape is nullptr.",
                       kTransData);
  auto output_dims = output_shape->GetDimSizes();
  auto output_format = output_shape->GetFormat();

  AttrValue *groups = ctx.GetAttr("groups");
  int64_t group = kGroupNum;
  if (groups != nullptr) {
      group = groups->GetInt();
     }
  KERNEL_LOG_INFO(
      "Begin trans formats from [%s] to [%s], shape [%s] to [%s], data type "
      "[%s] to [%s]",
      FormatToSerialString(input_format).c_str(),
      FormatToSerialString(output_format).c_str(),
      VectorToString(input_dims).c_str(), VectorToString(output_dims).c_str(),
      DTypeStr(input_data_type).c_str(), DTypeStr(output_data_type).c_str());
  const formats::TransArgs trans_args {
      input_data,
      static_cast<Format>(GetPrimaryFormat(input_format)),
      static_cast<Format>(GetPrimaryFormat(output_format)),
      input_dims,
      output_dims,
      input_data_type,
      group};
  if (input_data_type != output_data_type || input_dims.empty() ||
      !formats::FormatTransferExists(trans_args)) {
    KERNEL_LOG_WARN(
        "Transfer from format[%s] to [%s], shape [%s] to [%s], data type [%s] "
        "to [%s] is not supported",
        FormatToSerialString(input_format).c_str(),
        FormatToSerialString(output_format).c_str(),
        VectorToString(input_dims).c_str(), VectorToString(output_dims).c_str(),
        DTypeStr(input_data_type).c_str(), DTypeStr(output_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  formats::TransResult trans_result;
  auto ret = formats::TransFormat(trans_args, trans_result);
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_WARN(
        "Failed to trans formats from[%s] to [%s], shape [%s] to [%s], data "
        "type [%s]",
        FormatToSerialString(input_format).c_str(),
        FormatToSerialString(output_format).c_str(),
        VectorToString(input_dims).c_str(), VectorToString(output_dims).c_str(),
        DTypeStr(input_data_type).c_str());
    return ret;
  }

  auto output_data = output_tensor->GetData();
  auto output_length = output_tensor->GetDataSize();
  auto ret_mem = memcpy_s(output_data, output_length, trans_result.data.get(),
                          trans_result.length);
  if (ret_mem != EOK) {
    KERNEL_LOG_ERROR(
        "Memcpy from input[%llx]:size[%zu] to out[%llx]:size[%llu] failed, "
        "ret[%d].",
        trans_result.data.get(), trans_result.length, output_data,
        output_length, ret_mem);
    return KERNEL_STATUS_INNER_ERROR;
  }
  KERNEL_LOG_INFO(
      "End trans formats from [%s] to [%s], shape [%s] to [%s], data type "
      "[%s] to [%s]",
      FormatToSerialString(input_format).c_str(),
      FormatToSerialString(output_format).c_str(),
      VectorToString(input_dims).c_str(), VectorToString(output_dims).c_str(),
      DTypeStr(input_data_type).c_str(), DTypeStr(output_data_type).c_str());
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TransDataCpuKernel::DealData(T *input_data, T *output_data,
                                      Tensor *input_tensor,
                                      Tensor *output_tensor, int64_t group) {
  DataType dt = static_cast<DataType>(input_tensor->GetDataType());
  // if cube_k equals to DT_INT8, and let its values 32 else if equals to
  // DT_FLOAT16 or DT_FLOAT ,need to let its values 16. other dateType not
  // support compute.
  KERNEL_CHECK_FALSE(
      ((dt == DT_FLOAT16) || (dt == DT_INT8) || (dt == DT_FLOAT)),
      KERNEL_STATUS_PARAM_INVALID,
      "Input type is not DT_INT8 or DT_FLOAT16 or DT_FLOAT [%d]", dt);
  const int64_t cube_k = dt == DT_INT8 ? 32 : 16;
  auto input_shape = input_tensor->GetTensorShape();
  auto ge_input_format = input_shape->GetFormat();
  int32_t input_format = GetPrimaryFormat(ge_input_format);
  std::vector<int64_t> dims;
  dims = input_shape->GetDimSizes();
  int64_t d_dim = 0;
  int64_t h_dim = 0;
  int64_t w_dim = 0;
  int64_t c_dim = 0;
  int64_t n_dim = 0;
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
  } else {
    KERNEL_LOG_WARN(
        "Format is not FORMAT_DHWCN or FORMAT_NDHWC or FORMAT_NCDHW or "
        "FORMAT_NHWC or FORMAT_NCHW or FORMAT_HWCN, current input "
        "format is [%d]",
        input_format);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int64_t cin_ori = c_dim;
  int64_t cout_ori = n_dim / group;
  if (cin_ori == 0 || cout_ori == 0) {
    KERNEL_LOG_ERROR(
        "Cin_ori, cout_ori must not be equal 0, "
        "and current cin_ori, cout_ori, group are [%d][%d][%d]",
        cin_ori, cout_ori, group);
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
  (void)memset_s(output_data, sizeof(T) * size_output_data, 0,
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
              int64_t dst_inx =
                  (g / e_mult) * d_dim * c1_dim * h_dim * w_dim * cout_opt *
                      cube_k +
                  d * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                  (dst_ci / cube_k) * h_dim * w_dim * cout_opt * cube_k +
                  h * w_dim * cout_opt * cube_k + w * cout_opt * cube_k +
                  dst_co * cube_k + tempory;
              if ((input_format == FORMAT_DHWCN) ||
                  (input_format == FORMAT_HWCN)) {
                srx_inx = d * h_dim * w_dim * c_dim * n_dim +
                          h * w_dim * c_dim * n_dim + w * c_dim * n_dim +
                          c * n_dim + src_co;
              } else if ((input_format == FORMAT_NCDHW) ||
                         (input_format == FORMAT_NCHW)) {
                srx_inx = src_co * c_dim * d_dim * h_dim * w_dim +
                          c * d_dim * h_dim * w_dim + d * h_dim * w_dim +
                          h * w_dim + w;
              } else if ((input_format == FORMAT_NDHWC) ||
                         (input_format == FORMAT_NHWC)) {
                srx_inx = src_co * d_dim * h_dim * w_dim * c_dim +
                          d * h_dim * w_dim * c_dim + h * w_dim * c_dim +
                          w * c_dim + c;
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

// TransData supports input formats (NCDHW, DHWCN, NDHWC) convert to
// FORMAT_FRACTAL_Z_3D (GDC1HWN1N0C0), and also supports NHWC, NCHW, HWCN
// converte to FORMAT_FRACTAL_Z (GC1HWN1N0C0), HWCN to FZC04. The final effect
// achieved is for the data to be distributed diagonally. For example: When the
// input filter format is NCDHW, calculated the Correspondence of index between
// NCDHW and FORMAT_FRACTAL_Z_3D , then Convert the old filter to the new
// filter, and finally added 0 to the position where there is no data.
uint32_t TransDataCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "%s get input_tensor failed", kTransData);
  auto input_format = input_tensor->GetTensorShape()->GetFormat();
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "%s get output_tensor failed", kTransData);
  auto output_format = output_tensor->GetTensorShape()->GetFormat();
  if (!IsOriginSupportFormatTransfer(
          static_cast<Format>(GetPrimaryFormat(input_format)),
          static_cast<Format>(GetPrimaryFormat(output_format)))) {
    return NewCompute(ctx);
  }
  if ((input_format == FORMAT_HWCN) &&
      (output_format == FORMAT_FRACTAL_Z_C04)) {
    DataType data_type = static_cast<DataType>(input_tensor->GetDataType());
    int64_t cube = GetCubeSizeByDataType(data_type);
    if (cube < 0) {
      KERNEL_LOG_WARN("Don't support dtype[%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    uint64_t data_type_size = output_tensor->GetDataSize();
    uint64_t data_byte_size = GetSizeByDataType(data_type) * data_type_size;
    TransArgs args = {reinterpret_cast<uint8_t *>(input_tensor->GetData()),
                      input_tensor->GetTensorShape()->GetDimSizes(),
                      output_tensor->GetTensorShape()->GetDimSizes(),
                      data_type};
    auto output_addr = reinterpret_cast<uint8_t *>(output_tensor->GetData());
    uint32_t ret = FormatTransferHwcnToFZC04(args, output_addr, data_byte_size);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("FormatTransferHwcnToFZC04 function failed");
      return ret;
    }
    return KERNEL_STATUS_OK;
  }
  int32_t primary_out_put_format = GetPrimaryFormat(output_format);
  if ((primary_out_put_format != FORMAT_FRACTAL_Z) &&
      (primary_out_put_format != FORMAT_FRACTAL_Z_3D)) {
    KERNEL_LOG_EVENT("%s unsupport output_format [%d]", kTransData,
                     primary_out_put_format);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input_shape = input_tensor->GetTensorShape();
  std::vector<int64_t> dims;
  KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID,
                       "%s get input_shape failed", kTransData);
  dims = input_shape->GetDimSizes();
  if ((dims.size()) < 4) {
    KERNEL_LOG_WARN("%s dims size [%zu] must >= 4", kTransData, dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *groups = ctx.GetAttr("groups");
  int64_t group = kGroupNum;
  if (groups != nullptr) {
    group = groups->GetInt();
  }
  DataType dt = static_cast<DataType>(input_tensor->GetDataType());
  auto input_data_temp = input_tensor->GetData();
  KERNEL_CHECK_NULLPTR(input_data_temp, KERNEL_STATUS_PARAM_INVALID,
                       "%s get input_data failed", kTransData);
  auto output_data_temp = output_tensor->GetData();
  KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_PARAM_INVALID,
                       "%s get output_data failed", kTransData);
  uint32_t ret = 0;
  switch (dt) {
    case DT_INT8:
      ret = DealData(reinterpret_cast<int8_t *>(input_data_temp),
                     reinterpret_cast<int8_t *>(output_data_temp), input_tensor,
                     output_tensor, group);
      break;
    case DT_FLOAT:
      ret = DealData(reinterpret_cast<float *>(input_data_temp),
                     reinterpret_cast<float *>(output_data_temp), input_tensor,
                     output_tensor, group);
      break;
    case DT_FLOAT16:
      ret = DealData(reinterpret_cast<Eigen::half *>(input_data_temp),
                     reinterpret_cast<Eigen::half *>(output_data_temp),
                     input_tensor, output_tensor, group);
      break;

    default:
      KERNEL_LOG_WARN(
          "DateType is not DT_INT8 or DT_FLOAT or DT_FLOAT16, and current "
          "DataType is [%d]",
          dt);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

uint32_t TransDataCpuKernel::FormatTransferHwcnToFZC04(TransArgs &args,
                                                       uint8_t *output_addr,
                                                       uint64_t length) {
  KERNEL_LOG_DEBUG(
      "Begin to trans format from HWCN to FZC04, src shape [%s], data type "
      "[%s], dst shape [%s]",
      VectorToString(args.src_shape).c_str(),
      DTypeStr(args.src_data_type).c_str(),
      VectorToString(args.dst_shape).c_str());
  std::shared_ptr<uint8_t> dst_padding_one(nullptr);
  uint32_t ret = PaddingOne(args, dst_padding_one);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  std::vector<int64_t> perm_arg_tmp_one = {3, 0, 1, 2};
  std::shared_ptr<uint8_t> dst_transpose_one(nullptr);
  ret = Transpose(args, perm_arg_tmp_one, dst_transpose_one);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  std::vector<int64_t> &src_shape = args.src_shape;
  std::vector<int64_t> src_shape_tmp = src_shape;
  src_shape.resize(2);
  src_shape[0] = src_shape_tmp[0];
  src_shape[1] = src_shape_tmp[1] * src_shape_tmp[2] * src_shape_tmp[3];
  std::shared_ptr<uint8_t> dst_padding_two(nullptr);
  ret = PaddingTwo(args, dst_padding_two);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  int64_t cube = GetCubeSizeByDataType(args.src_data_type);
  src_shape_tmp = src_shape;
  src_shape.resize(4);
  src_shape[0] = src_shape_tmp[0] / cube;
  src_shape[1] = cube;
  src_shape[2] = src_shape_tmp[1] / cube;
  src_shape[3] = cube;
  std::vector<int64_t> perm_arg_tmp_two = {2, 0, 1, 3};
  std::shared_ptr<uint8_t> dst_transpose_two(nullptr);
  ret = Transpose(args, perm_arg_tmp_two, dst_transpose_two);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  ret = memcpy_s(
      output_addr, length, args.data,
      VectorToNum(args.src_shape) * GetSizeByDataType(args.src_data_type));
  if (ret != 0) {
    KERNEL_LOG_ERROR("Memcpy failed, ret is [%d]", ret);
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::PaddingOne(TransArgs &args,
                                        std::shared_ptr<uint8_t> &dst) {
  DataType data_type = args.src_data_type;
  std::vector<int64_t> dst_shape;
  uint32_t ret = GetPaddingOneShape(args, dst_shape);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  int32_t type_size = GetSizeByDataType(data_type);
  int64_t dst_byte_size = VectorToNum(dst_shape) * type_size;
  dst.reset(new (std::nothrow) uint8_t[dst_byte_size],
            std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    KERNEL_LOG_ERROR("New Memory failed!");
    return KERNEL_STATUS_INNER_ERROR;
  }
  auto ret_mem = memset_s(dst.get(), dst_byte_size, 0, dst_byte_size);
  if (ret_mem != 0) {
    KERNEL_LOG_ERROR("Memst failed, ret is [%d]", ret_mem);
    return KERNEL_STATUS_INNER_ERROR;
  }
  std::vector<int64_t> &src_shape = args.src_shape;
  auto h = src_shape.at(0);
  auto w = src_shape.at(1);
  auto c = src_shape.at(2);
  auto n = src_shape.at(3);
  auto h_padding = dst_shape[0];
  auto w_padding = dst_shape[1];
  auto c_padding = dst_shape[2];
  auto n_padding = dst_shape[3];
  auto src_add = args.data;
  auto dst_add = dst.get();
  auto protect_size = h_padding * w_padding * c_padding * n_padding * type_size;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < c; k++) {
        auto dst_stride =
            ((i * w_padding + j) * c_padding + k) * n_padding * type_size;
        auto ret_cpy = memcpy_s(dst_add + dst_stride, protect_size - dst_stride,
                                src_add + ((i * w + j) * c + k) * n * type_size,
                                n * type_size);
        if (ret_cpy != 0) {
          KERNEL_LOG_ERROR("Memcpy failed, ret is [%d]", ret_cpy);
          return KERNEL_STATUS_INNER_ERROR;
        }
      }
    }
  }
  args.data = dst.get();
  src_shape = dst_shape;
  return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::PaddingTwo(TransArgs &args,
                                        std::shared_ptr<uint8_t> &dst) {
  DataType data_type = args.src_data_type;
  std::vector<int64_t> dst_shape;
  uint32_t ret = GetPaddingTwoShape(args, dst_shape);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  int32_t type_size = GetSizeByDataType(data_type);
  int64_t dst_byte_size = VectorToNum(dst_shape) * type_size;
  dst.reset(new (std::nothrow) uint8_t[dst_byte_size],
            std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    KERNEL_LOG_ERROR("New Memory failed!");
    return KERNEL_STATUS_INNER_ERROR;
  }
  auto ret_mem = memset_s(dst.get(), dst_byte_size, 0, dst_byte_size);
  if (ret_mem != 0) {
    KERNEL_LOG_ERROR("Memst failed, ret is [%d]", ret_mem);
    return KERNEL_STATUS_INNER_ERROR;
  }
  std::vector<int64_t> &src_shape = args.src_shape;
  auto n = src_shape.at(0);
  auto z = src_shape.at(1);
  auto n_padding = dst_shape[0];
  auto z_padding = dst_shape[1];
  auto src_add = args.data;
  auto dst_add = dst.get();
  auto protect_size = n_padding * z_padding * type_size;
  for (int i = 0; i < n; i++) {
    auto dst_stride = i * z_padding * type_size;
    auto ret_cpy = memcpy_s(dst_add + dst_stride, protect_size - dst_stride,
                            src_add + i * z * type_size, z * type_size);
    if (ret_cpy != 0) {
      KERNEL_LOG_ERROR("Memcpy failed, ret is [%d]", ret_cpy);
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  args.data = dst.get();
  src_shape = dst_shape;
  return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::GetPaddingOneShape(
    const TransArgs &args, std::vector<int64_t> &dst_shape) {
  int64_t cube = GetCubeSizeByDataType(args.src_data_type);
  auto h = args.src_shape.at(0);
  auto w = args.src_shape.at(1);
  auto c = args.src_shape.at(2);
  auto n = args.src_shape.at(3);
  if (c > kMaxDimsNumC) {
    KERNEL_LOG_ERROR("Invalid dim c num[%lu].It should be in (0, %ld]", c,
                     kMaxDimsNumC);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  dst_shape.resize(4);
  dst_shape[0] = h;
  dst_shape[1] = w;
  dst_shape[2] = kMaxDimsNumC;
  int64_t tmp = Ceil(n, cube);
  dst_shape[3] = tmp * cube;
  return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::GetPaddingTwoShape(
    const TransArgs &args, std::vector<int64_t> &dst_shape) {
  int64_t cube = GetCubeSizeByDataType(args.src_data_type);
  auto n = args.src_shape.at(0);
  auto z = args.src_shape.at(1);
  dst_shape.resize(2);
  dst_shape[0] = n;
  int64_t tmp = Ceil(z, cube);
  dst_shape[1] = tmp * cube;
  return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::Transpose(TransArgs &args,
                                       const std::vector<int64_t> &perm_arg,
                                       std::shared_ptr<uint8_t> &dst) {
  std::vector<int64_t> &src_shape = args.src_shape;
  std::vector<int64_t> dst_shape;
  TransShapeByPerm(src_shape, perm_arg, dst_shape);
  DataType src_data_type = args.src_data_type;
  KERNEL_LOG_DEBUG(
      "Begin to transpose, src shape [%s], perm arg [%s], dst shape [%s], data "
      "type [%s]",
      VectorToString(src_shape).c_str(), VectorToString(perm_arg).c_str(),
      VectorToString(dst_shape).c_str(), DTypeStr(src_data_type).c_str());
  int64_t dst_ele_num = VectorToNum(dst_shape);
  int64_t data_size = GetSizeByDataType(src_data_type);
  int64_t dst_size = data_size * dst_ele_num;
  dst.reset(new (std::nothrow) uint8_t[dst_size],
            std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    KERNEL_LOG_ERROR("New Memory failed!");
    return KERNEL_STATUS_INNER_ERROR;
  }
  int64_t dst_index = 0;
  std::vector<int64_t> src_shape_head;
  GetShapeHead(src_shape, src_shape_head);
  std::vector<int64_t> dst_shape_head;
  GetShapeHead(dst_shape, dst_shape_head);
  std::map<int32_t, int32_t> index_map;
  GetIndexMap(perm_arg, index_map);
  while (dst_index < dst_ele_num) {
    auto src_index = GetSrcIndex(dst_index, src_shape, dst_shape,
                                 src_shape_head, dst_shape_head, index_map);
    auto ret = memcpy_s(dst.get() + dst_index * data_size,
                        dst_size - dst_index * data_size,
                        args.data + src_index * data_size, data_size);
    if (ret != 0) {
      KERNEL_LOG_ERROR("Memcpy failed, ret is [%d]", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
    dst_index += 1;
  }
  src_shape = dst_shape;
  args.data = dst.get();
  return KERNEL_STATUS_OK;
}

int64_t TransDataCpuKernel::GetCubeSizeByDataType(DataType data_type) {
  // Current cube does not support 4 bytes and longer data
  auto size = GetSizeByDataType(data_type);
  if (size <= 0) {
    KERNEL_LOG_ERROR("Failed to get cube size, the data type [%s] is invalid",
                     DTypeStr(data_type).c_str());
    return -1;
  } else if (size == 1) {
    return kCubeSize * 2;  // 32 bytes cube size
  } else {
    return kCubeSize;
  }
}
REGISTER_CPU_KERNEL(kTransData, TransDataCpuKernel);
}  // namespace aicpu