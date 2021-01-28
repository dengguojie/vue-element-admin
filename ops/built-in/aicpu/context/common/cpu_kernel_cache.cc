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
#include "cpu_kernel_cache.h"

#include <limits.h>

#include "cce/aicpu_engine_struct.h"
#include "cpu_kernel.h"
#include "cpu_kernel_register.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"

using namespace aicpu;

namespace {
// max io address number limit is 1024
constexpr uint32_t kMaxIoAddrNumParamLen = 1024;
// max LRU cache number is 256
constexpr uint32_t kMaxLRUCacheNum = 256;
}  // namespace

namespace aicpu {
CpuKernelCache::CpuKernelCache()
    : unknown_shape_(false),
      run_dynamic_(true),
      nodedef_(nullptr),
      nodedef_len_(0),
      nodedef_proto_(nullptr) {}

/*
 * Init kernel cache.
 */
int32_t CpuKernelCache::InitParameter() {
  if (!GetSessionFlag()) {
    SetCapacity(kMaxLRUCacheNum);
  }
  return 0;
}

/*
 * update framework output tensor shape.
 */
uint32_t CpuKernelCache::UpdateFWKOutputShape(const CpuKernelContext &ctx) {
  if (unknown_shape_) {
    for (size_t i = 0; i < ctx.GetOutputsSize(); ++i) {
      Tensor *output = ctx.Output(i);
      KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID,
                           "Get output[%zu] failed.", i)
      auto shape = output->GetTensorShape();
      KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID,
                           "Get output[%zu] shape failed.", i)

      for (int32_t index = 0; index < shape->GetDims(); ++index) {
        output_shape_and_type_[i]->dims[index] = shape->GetDimSize(index);
      }
    }
  }
  return KERNEL_STATUS_OK;
}

/*
 * get shape information from framework.
 */
void CpuKernelCache::GetDimsFromShapeAndType(
    const FWKAdapter::ShapeAndType *shape_and_type,
    std::vector<int64_t> &dims) {
  for (uint32_t index = 0; index < FWKAdapter::kMaxShapeDims; ++index) {
    // LLONG_MIN for dim end flag
    if (shape_and_type->dims[index] == LLONG_MIN) {
      break;
    }
    int64_t dim_value = shape_and_type->dims[index];
    KERNEL_LOG_INFO("Get extend shape[%u] is [%lld]", index, dim_value);
    dims.emplace_back(dim_value);
  }
}

/*
 * update tensor information.
 */
uint32_t CpuKernelCache::UpdateTensor(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Update tensor info begin.");
  if (io_addrs_.size() != ctx.GetInputsSize() + ctx.GetOutputsSize()) {
    KERNEL_LOG_ERROR(
        "Addr number[%zu] is not equal to the sum of inputs[%zu] and "
        "output[%zu].",
        io_addrs_.size(), ctx.GetInputsSize(), ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ((unknown_shape_) &&
      ((input_shape_and_type_.size() != ctx.GetInputsSize()) ||
       (output_shape_and_type_.size() != ctx.GetOutputsSize()))) {
    KERNEL_LOG_ERROR(
        "Input shape_and_type size error, input size[%zu], input "
        "shape_and_type "
        "size[%zu], output size[%zu], output shape_and_type size[%zu].",
        ctx.GetInputsSize(), input_shape_and_type_.size(), ctx.GetOutputsSize(),
        output_shape_and_type_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  size_t addr_index = 0;
  for (size_t i = 0; i < ctx.GetInputsSize(); ++i, ++addr_index) {
    Tensor *input = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_PARAM_INVALID,
                         "Get input[%zu] failed.", i)
    input->SetData(reinterpret_cast<void *>(
        static_cast<uintptr_t>(io_addrs_[addr_index])));
    int64_t calc_data_size = input->CalcDataSizeByShape();
    uint64_t data_size = calc_data_size < 0 ? 0 : calc_data_size;
    input->SetDataSize(data_size);
    KERNEL_LOG_INFO("Set input[%zu] addr[%llu] success.", i,
                    io_addrs_[addr_index]);

    if (unknown_shape_) {
      std::vector<int64_t> dims;
      GetDimsFromShapeAndType(input_shape_and_type_[i], dims);
      auto shape = input->GetTensorShape();
      KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID,
                           "Get input[%zu] shape failed.", i)
      shape->SetDimSizes(dims);
    }
  }

  for (size_t i = 0; i < ctx.GetOutputsSize(); i++, addr_index++) {
    Tensor *output = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID,
                         "Get output[%zu] failed.", i)
    output->SetData(reinterpret_cast<void *>(
        static_cast<uintptr_t>(io_addrs_[addr_index])));
    int64_t calc_data_size = output->CalcDataSizeByShape();
    uint64_t data_size = calc_data_size < 0 ? 0 : calc_data_size;
    output->SetDataSize(data_size);
    KERNEL_LOG_INFO("Set output[%zu] addr[%llu] success.", i,
                    io_addrs_[addr_index]);

    if (unknown_shape_) {
      std::vector<int64_t> dims;
      GetDimsFromShapeAndType(output_shape_and_type_[i], dims);
      auto shape = output->GetTensorShape();
      KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID,
                           "Get output[%zu] shape failed.", i)
      shape->SetDimSizes(dims);
    }
  }
  KERNEL_LOG_INFO("Update tensor info success.");
  return KERNEL_STATUS_OK;
}

/*
 * parse extend tensor shape types information.
 */
uint32_t CpuKernelCache::ParseExtShapeType(
    const FWKAdapter::ExtInfo *ext_info) {
  if (ext_info->infoLen != sizeof(int32_t)) {
    KERNEL_LOG_ERROR(
        "Parse extend shape type failed, as info length must be [%zu], but got "
        "[%u].",
        sizeof(int32_t), ext_info->infoLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  unknown_shape_ = true;
  KERNEL_LOG_INFO("Kernel has unknown shape.");
  return KERNEL_STATUS_OK;
}

/*
 * parse extend tensor shape and types information.
 */
uint32_t CpuKernelCache::ParseExtShapeAndType(
    FWKAdapter::ExtInfo *ext_info,
    std::vector<FWKAdapter::ShapeAndType *> &shape_and_type) {
  if (!run_dynamic_) {
    return KERNEL_STATUS_OK;
  }
  shape_and_type.clear();
  uint32_t size = (ext_info->infoLen) / sizeof(FWKAdapter::ShapeAndType);
  KERNEL_LOG_INFO("Parse extend shape and type, size[%u].", size);
  uint32_t check = (ext_info->infoLen) % sizeof(FWKAdapter::ShapeAndType);
  if (check != 0) {
    KERNEL_LOG_ERROR(
        "Parse extend info length[%u] failed, must be integer multiple of the "
        "[%zu].",
        ext_info->infoLen, sizeof(FWKAdapter::ShapeAndType));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto shapes = reinterpret_cast<FWKAdapter::ShapeAndType *>(ext_info->infoMsg);
  for (uint32_t index = 0; index < size; ++index) {
    shape_and_type.emplace_back(&shapes[index]);
  }
  return KERNEL_STATUS_OK;
}

/*
 * parse extend session information.
 */
uint32_t CpuKernelCache::ParseExtSessionInfo(FWKAdapter::ExtInfo *ext_info,
                                             uint64_t &kernel_id) {
  // no overflow
  KERNEL_LOG_INFO("Parse extend session info.");
  auto need_len = sizeof(SessionInfo);
  if (ext_info->infoLen != need_len) {
    KERNEL_LOG_ERROR(
        "Parse extend session info failed, as info length must be "
        "[%zu], but got [%u].",
        sizeof(SessionInfo), ext_info->infoLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto session = reinterpret_cast<SessionInfo *>(ext_info->infoMsg);
  kernel_id = session->kernelId;
  return KERNEL_STATUS_OK;
}

/*
 * get bit status.
 */
bool CpuKernelCache::GetBitStatus(int num, int pos) {
  return ((num & (1 << pos)) != 0);
}

/*
 * parse bitmap information.
 */
uint32_t CpuKernelCache::ParseExtBitMap(const FWKAdapter::ExtInfo *ext_info) {
  if (ext_info->infoLen != sizeof(int64_t)) {
    KERNEL_LOG_ERROR(
        "Parse extend bitmap failed, as info length must be [%zu], but got "
        "[%u].",
        sizeof(int64_t), ext_info->infoLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int64_t bit_map = *(reinterpret_cast<const int64_t *>(ext_info->infoMsg));
  run_dynamic_ = (!GetBitStatus(bit_map, 0));
  unknown_shape_ = run_dynamic_;
  KERNEL_LOG_INFO("Run_dynamic_ is [%d].", run_dynamic_);
  return KERNEL_STATUS_OK;
}

/*
 * parse extend information.
 */
uint32_t CpuKernelCache::ParseExtMsg(AicpuParamHead *param_head,
                                     bool &has_session_info,
                                     uint64_t &kernel_id) {
  KERNEL_LOG_INFO("Parse extend info and update shape begin.");
  unknown_shape_ = false;
  uint32_t offset = 0;
  FWKAdapter::ExtInfo *ext_info = nullptr;
  char *extInfo_buf =
      reinterpret_cast<char *>(static_cast<uintptr_t>(param_head->extInfoAddr));
  while (offset + sizeof(FWKAdapter::ExtInfo) <= param_head->extInfoLength) {
    ext_info = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfo_buf + offset);
    if (ext_info == nullptr) {
      KERNEL_LOG_ERROR(
          "Extend info is nullptr, extInfo length[%u], extend info addr[%p], "
          "offset[%u].",
          param_head->extInfoLength, param_head->extInfoAddr, offset);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    uint32_t ret = KERNEL_STATUS_OK;
    switch (ext_info->infoType) {
      case FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        ret = ParseExtShapeType(ext_info);
        break;
      case FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        ret = ParseExtShapeAndType(ext_info, input_shape_and_type_);
        break;
      case FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        ret = ParseExtShapeAndType(ext_info, output_shape_and_type_);
        break;
      case FWKAdapter::FWK_ADPT_EXT_SESSION_INFO:
        has_session_info = true;
        ret = ParseExtSessionInfo(ext_info, kernel_id);
        break;
      case FWKAdapter::FWK_ADPT_EXT_BITMAP:
        ret = ParseExtBitMap(ext_info);
        break;
      default:
        KERNEL_LOG_INFO("Ignore infoType[%d], infoLen[%u].", ext_info->infoType,
                        ext_info->infoLen);
        break;
    }

    if (ret != KERNEL_STATUS_OK) {
      return ret;
    }

    // not overflow
    offset += FWKAdapter::kExtInfoHeadSize;
    offset += ext_info->infoLen;
  }

  return KERNEL_STATUS_OK;
}

/*
 * parse io address.
 */
uint32_t CpuKernelCache::ParseIoAddr(AicpuParamHead *param_head) {
  auto param_base = reinterpret_cast<char *>(param_head);
  char *extend_param_base = param_base + sizeof(AicpuParamHead);
  uint32_t extend_param_len = param_head->length - sizeof(AicpuParamHead);
  io_addrs_.clear();

  if (param_head->ioAddrNum > 0) {
    if (param_head->ioAddrNum > kMaxIoAddrNumParamLen) {
      KERNEL_LOG_ERROR("Param ioAddrNum[%u] is over %u.", param_head->ioAddrNum,
                       kMaxIoAddrNumParamLen);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    uint32_t addr_len = param_head->ioAddrNum * sizeof(uint64_t);
    if (extend_param_len < addr_len) {
      KERNEL_LOG_ERROR(
          "Extend param is not enough for io addr, ioAddrNum[%u], "
          "extend_param_len[%u].",
          param_head->ioAddrNum, extend_param_len);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    auto io_addr_base = reinterpret_cast<uint64_t *>(extend_param_base);
    for (uint32_t i = 0; i < param_head->ioAddrNum; ++i) {
      io_addrs_.push_back(io_addr_base[i]);
    }
    extend_param_base = extend_param_base + addr_len;
    extend_param_len -= addr_len;
  }

  if (extend_param_len < sizeof(uint32_t)) {
    KERNEL_LOG_ERROR(
        "Extend param is not enough for addr, needLen[%zu], "
        "extend_param_len[%u].",
        sizeof(uint32_t), extend_param_len);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  nodedef_len_ = *reinterpret_cast<uint32_t *>(extend_param_base);
  extend_param_base += sizeof(uint32_t);
  nodedef_ = extend_param_base;
  KERNEL_LOG_INFO("Parse io addr success, io number[%zu], nodedef length[%u].",
                  io_addrs_.size(), nodedef_len_);
  return KERNEL_STATUS_OK;
}

/*
 * get cpu kernel context from cache
 */
std::shared_ptr<CpuKernelContext> CpuKernelCache::GetCpuKernelContext(
    bool has_sess_info, uint64_t kernel_id) {
  std::shared_ptr<CpuKernelContext> ctx = nullptr;
  KERNEL_LOG_INFO("Get cpu kernel context begin, kernel id[%llu].", kernel_id);
  if (has_sess_info) {
    CpuCacheData *cache = GetCache(kernel_id);
    if (cache != nullptr) {
      KERNEL_LOG_INFO("Get kernel from cache success.");
      return cache->context;
    }
  }

  std::string str_data(nodedef_, nodedef_len_);
  nodedef_proto_ = CpuKernelUtils::CreateNodeDef();
  KERNEL_CHECK_NULLPTR(nodedef_proto_,
                       std::shared_ptr<CpuKernelContext>(nullptr),
                       "Create node def failed.")
  if (!nodedef_proto_->ParseFromString(str_data)) {
    return std::shared_ptr<CpuKernelContext>(nullptr);
  }

  CpuKernelContext *tmp = new (std::nothrow) CpuKernelContext(DEVICE);
  KERNEL_CHECK_NULLPTR(tmp, std::shared_ptr<CpuKernelContext>(nullptr),
                       "Create context failed.")
  ctx = std::shared_ptr<CpuKernelContext>(tmp);
  uint32_t ret = ctx->Init(nodedef_proto_.get());
  if (ret != KERNEL_STATUS_OK) {
    return std::shared_ptr<CpuKernelContext>(nullptr);
  }

  if (has_sess_info) {
    CpuCacheData *cache_ptr =
        new (std::nothrow) CpuCacheData(nodedef_proto_, ctx);
    KERNEL_CHECK_NULLPTR(cache_ptr, std::shared_ptr<CpuKernelContext>(nullptr),
                         "Create cpu cache data failed.")
    std::shared_ptr<CpuCacheData> cache_shared =
        std::shared_ptr<CpuCacheData>(cache_ptr);
    SetCache(kernel_id, cache_shared);
    KERNEL_LOG_INFO("Cache cpu kernel data success, kernel id[%llu].",
                    kernel_id);
  }
  KERNEL_LOG_INFO("Get cpu kernel context success, kernel id[%llu].",
                  kernel_id);
  return ctx;
}

/*
 * run kernel.
 */
int32_t CpuKernelCache::RunKernel(void *param) {
  AicpuParamHead *param_head = static_cast<AicpuParamHead *>(param);
  uint32_t ret = ParseIoAddr(param_head);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  bool has_sess_info = false;
  uint64_t kernel_id = 0;
  run_dynamic_ = true;
  ret = ParseExtMsg(param_head, has_sess_info, kernel_id);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  auto ctx = GetCpuKernelContext(has_sess_info, kernel_id);
  KERNEL_CHECK_NULLPTR(ctx, KERNEL_STATUS_INNER_ERROR,
                       "Get cpu kernel context from buff failed.")

  ret = UpdateTensor(*ctx);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  ret = CpuKernelRegister::Instance().RunCpuKernel(*ctx);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  ret = UpdateFWKOutputShape(*ctx);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }
  return 0;
}

}  // namespace aicpu
