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

#include "format_transfer_hwcn_fractal_zn_lstm.h"

#include "format_transfer_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"


namespace aicpu {
namespace formats {
namespace {

const size_t kHwcnH = 0;
const size_t kHwcnW = 1;
const size_t kHwcnC = 2;
const size_t kHwcnN = 3;
const size_t kHwcnDimsNum = 4;
const size_t kFracZDimsNum = 4;

bool CheckDataTypeSupported(const DataType &data_type) {
  return (data_type == DT_FLOAT || data_type == DT_FLOAT16);
}

bool IsShapeArgValid(const std::vector<int64_t> &src_shape, const std::vector<int64_t> &perm_arg) {
  if (src_shape.empty()) {
    std::string error = "Failed to transpose, empty src shape";
    KERNEL_LOG_ERROR("[Trans][Shape]Failed, empty src shape");
    return false;
  }
  for (auto dim : src_shape) {
    if (dim < 0) {
      std::string error = "Failed to transpose, negative dim in src shape " + FmtToStr(VectorToString(src_shape));
      KERNEL_LOG_ERROR("%s", error.c_str());
      return false;
    }
  }
  if (perm_arg.size() != src_shape.size()) {
    std::string error = "Failed to transpose, the size of src shape" + FmtToStr(src_shape.size()) +
        " and perm arg" +  FmtToStr(perm_arg.size()) + " are different";
    KERNEL_LOG_ERROR("%s", error.c_str());
    return false;
  }

  std::vector<int64_t> exists(perm_arg.size());
  for (auto perm : perm_arg) {
    if (perm < 0 || static_cast<size_t>(perm) >= perm_arg.size() || ++exists[perm] > 1) {
      std::string error = "Failed to transpose, duplicated perm arg " + FmtToStr(perm) +
        ", perm arg " +  FmtToStr(VectorToString(perm_arg));
      KERNEL_LOG_ERROR("%s", error.c_str());
      return false;
    }
  }
  return true;
}
bool IsTransposeArgValid(const uint8_t *src, const std::vector<int64_t> &src_shape, DataType src_data_type,
                         const std::vector<int64_t> &perm_arg) {
  if (src == nullptr) {
    KERNEL_LOG_ERROR("[Trans][Param]Failed, the src is null");
    return false;
  }
  if (GetSizeByDataType(src_data_type) < 0) {
    KERNEL_LOG_ERROR("[Trans][Param]Failed, the data type %s is not support",
           DTypeStr(src_data_type).c_str());
    return false;
  }
  return IsShapeArgValid(src_shape, perm_arg);
}

std::vector<int64_t> GenHeads(const std::vector<int64_t> &shape) {
  std::vector<int64_t> heads(shape.size());
  bool first = true;
  for (auto i = static_cast<int64_t>(shape.size() - 1); i >= 0; --i) {
    if (first) {
      heads[i] = 1;
      first = false;
    } else {
      heads[i] = shape[i + 1] * heads[i + 1];
    }
  }
  return heads;
}

int64_t GenOffset(const std::vector<int64_t> &offsets, const std::vector<int64_t> &indexes) {
  int64_t offset = 0;
  for (size_t i = 0; i < indexes.size(); ++i) {
    offset += offsets[i] * indexes[i];
  }
  return offset;
}

void AddOne(const std::vector<int64_t> &shape, std::vector<int64_t> &indexes) {
  size_t i = indexes.size() - 1;
  indexes[i]++;
  while (i > 0) {
    if (indexes[i] >= shape[i]) {
      indexes[i] = 0;
      indexes[i - 1]++;
      --i;
    } else {
      break;
    }
  }
}

std::vector<int64_t> TransShapeByPerm(const std::vector<int64_t> &src_shape, const std::vector<int64_t> &perm_arg) {
  std::vector<int64_t> dst_shape(src_shape.size());
  for (size_t i = 0; i < perm_arg.size(); ++i) {
    dst_shape[i] = src_shape[perm_arg[i]];
  }
  return dst_shape;
}


uint32_t Transpose(const uint8_t *src, const std::vector<int64_t> &src_shape, DataType src_data_type,
                 const std::vector<int64_t> &perm_arg, TransResult &result) {
  if (!IsTransposeArgValid(src, src_shape, src_data_type, perm_arg)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto dst_shape = TransShapeByPerm(src_shape, perm_arg);
  auto src_origin_ordered_heads = GenHeads(src_shape);
  auto src_heads = TransShapeByPerm(src_origin_ordered_heads, perm_arg);

  int64_t dst_ele_num = GetItemNumByShape(dst_shape);
  int64_t data_size = GetSizeByDataType(src_data_type);
  int64_t dst_size = data_size * dst_ele_num;

  KERNEL_LOG_DEBUG("Begin to transpose, src shape %s, perm arg %s, dst shape %s, data type %s", VectorToString(src_shape).c_str(),
         VectorToString(perm_arg).c_str(), VectorToString(dst_shape).c_str(),
         DTypeStr(src_data_type).c_str());
  if (dst_ele_num == 0) {
    result.length = static_cast<size_t>(dst_size);
    return KERNEL_STATUS_OK;
  }

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  int64_t dst_index = 0;
  std::vector<int64_t> dst_indexes(dst_shape.size());
  while (dst_index < dst_ele_num) {
    auto src_offset = GenOffset(src_heads, dst_indexes) * data_size;
    auto dst_offset_bytes = dst_index * data_size;
    auto protected_size = dst_size - dst_offset_bytes < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                              ? dst_size - dst_offset_bytes
                              : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
    auto ret = memcpy_s(dst.get() + dst_offset_bytes, static_cast<size_t>(protected_size), src + src_offset,
                        static_cast<size_t>(data_size));
    if (ret != EOK) {
      KERNEL_LOG_ERROR(
             "[Operate][Memory]Failed to transpose, src shape %s, perm arg %s, dst shape %s, "
             "failed to write to dst offset %ld, current dim offset %s",
             VectorToString(src_shape).c_str(), VectorToString(perm_arg).c_str(), VectorToString(dst_shape).c_str(),
             dst_offset_bytes, VectorToString(dst_indexes).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    AddOne(dst_shape, dst_indexes);
    ++dst_index;
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return KERNEL_STATUS_OK;
}


uint32_t TransShapeHwcnToFrazlstm(const DataType &data_type, const std::vector<int64_t> &src_shape,
                                 std::vector<int64_t> &dst_shape) {
  auto cube_size = GetCubeSizeByDataType(data_type);
  dst_shape.clear();
  dst_shape.push_back(Ceil(src_shape.at(kHwcnC), static_cast<int64_t>(cube_size)));
  dst_shape.push_back(Ceil(src_shape.at(kHwcnN), static_cast<int64_t>(cube_size)));
  dst_shape.push_back(cube_size);
  dst_shape.push_back(cube_size);
  if (!CheckShapeValid(dst_shape, kFracZDimsNum)) {
    KERNEL_LOG_ERROR("Failed to check dst shape %s",
           VectorToString(dst_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CheckArgsForHwcnToFrazlstm(const TransArgs &args) {
  if (args.src_format != FORMAT_HWCN || args.dst_format != FORMAT_FRACTAL_ZN_LSTM) {
    std::string error = "Dose not support trans format from " +
        FmtToStr(FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(FormatToSerialString(args.dst_format));
    KERNEL_LOG_ERROR("%s", error.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (!CheckDataTypeSupported(args.src_data_type)) {
    KERNEL_LOG_ERROR("Failed to trans shape from HWCN to FRACTAL_ZN_LSTM, invalid data type %s",
           DTypeStr(args.src_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (!CheckShapeValid(args.src_shape, kHwcnDimsNum)) {
    KERNEL_LOG_ERROR("Failed to check src shape %s", VectorToString(args.src_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (!CheckShapeValid(args.dst_shape, kFracZDimsNum)) {
    KERNEL_LOG_ERROR("Failed to check dst shape %s", VectorToString(args.dst_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> expect_dst_shape;
  auto ret = TransShapeHwcnToFrazlstm(args.src_data_type, args.src_shape, expect_dst_shape);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  if (args.dst_shape != expect_dst_shape) {
    KERNEL_LOG_ERROR(
           "Failed to trans format, src and dst shape are not compatible. src shape %s, dst shape %s, "
           "expect dst shape %s",
           VectorToString(args.src_shape).c_str(), VectorToString(args.dst_shape).c_str(),
           VectorToString(expect_dst_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t GetDstDataAfterTrans(const TransArgs &args, TransResult &result, const int size, const int64_t total_size) {
  int64_t axis_n =  args.src_shape[args.src_shape.size() - 1] / 4;
  int64_t axis_ni = 16;
  int64_t axis_c0 = 16;
  int64_t axis_no = (axis_n +  axis_ni - 1) / axis_ni;
  int64_t axis_c_p1 = args.src_shape[args.src_shape.size() - 2] - axis_n;
  int64_t axis_c_p2 = axis_n;
  int64_t axis_c1_p1 = (axis_c_p1 +  axis_c0 - 1) / axis_c0;
  int64_t axis_c1_p2 = (axis_c_p2 +  axis_c0 - 1) / axis_c0;
  int64_t axis_hw = 1;

  const std::vector<int64_t> src_shape = {axis_hw, axis_c1_p1 + axis_c1_p2, axis_c0, 4 * axis_no, axis_ni};
  const std::vector<int64_t> perm_arg = {1, 0, 3, 4, 2};

  KERNEL_LOG_DEBUG("Begin to transpose, src shape %s, perm arg %s", VectorToString(src_shape).c_str(),
        VectorToString(perm_arg).c_str());
  if (Transpose(args.data, src_shape, args.src_data_type, perm_arg, result) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR(
           "Failed to trans format, src and dst shape are not compatible. src shape %s, dst shape %s. ",
           VectorToString(src_shape).c_str(), VectorToString(perm_arg).c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }

  return KERNEL_STATUS_OK;
}
}  // namespace

uint32_t FormatTransferHwcnFractalznlstm::TransFormat(const TransArgs &args, TransResult &result) {
  if (CheckArgsForHwcnToFrazlstm(args) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int size = GetSizeByDataType(args.src_data_type);
  auto total_size = GetItemNumByShape(args.dst_shape) * size;
  if (total_size <= 0) {
    int64_t src_size = GetItemNumByShape(args.src_shape);
    if (total_size == 0 && src_size == 0) {
      result.length = static_cast<size_t>(total_size);
      return KERNEL_STATUS_OK;
    }

    KERNEL_LOG_ERROR("Get %ld total size from dst shape %s, src shape %s", total_size,
           VectorToString(args.dst_shape).c_str(), VectorToString(args.src_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_LOG_DEBUG("Begin to trans format from HWCN to Fractal_Zn_Lstm, src shape %s, data type %s, dst shape %s, memory size %ld",
         VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
         VectorToString(args.dst_shape).c_str(), total_size);
  if (GetDstDataAfterTrans(args, result, size, total_size) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Failed to get data after trans, src shape %s, data type %s, dst shape %s, memory size %ld",
           VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
           VectorToString(args.dst_shape).c_str(), total_size);
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t FormatTransferHwcnFractalznlstm::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                               DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape, int64_t groups) {
  if (src_format == FORMAT_HWCN && CheckDataTypeSupported(data_type)) {
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
      KERNEL_LOG_ERROR("Failed to check src shape %s",
             VectorToString(src_shape).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    return TransShapeHwcnToFrazlstm(data_type, src_shape, dst_shape);
  } else if (src_format != FORMAT_HWCN) {
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_FORMAT_TRANSFER(FormatTransferHwcnFractalznlstm, FORMAT_HWCN, FORMAT_FRACTAL_ZN_LSTM)
}  // namespace formats
}  // namespace ge
