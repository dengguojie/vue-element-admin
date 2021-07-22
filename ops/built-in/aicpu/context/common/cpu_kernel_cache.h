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
#ifndef AICPU_CPU_KERNEL_CACHE_H_
#define AICPU_CPU_KERNEL_CACHE_H_

#include <map>
#include <memory>

#include "aicpu_task_struct.h"
#include "cce/fwk_adpt_struct.h"
#include "cpu_context.h"
#include "cpu_node_def.h"
#include "kernel_cache.h"

namespace aicpu {
struct CpuCacheData {
  std::shared_ptr<NodeDef> proto = nullptr;
  std::shared_ptr<CpuKernelContext> context = nullptr;
  CpuCacheData(std::shared_ptr<NodeDef> proto,
               std::shared_ptr<CpuKernelContext> context)
      : proto(proto), context(context) {}
};

class CpuKernelCache : public KernelCache<CpuCacheData> {
 public:
  CpuKernelCache() = default;
  ~CpuKernelCache() = default;

  /*
   * Init kernel cache.
   * @return int32_t: 0 indicates success, while the others fail
   */
  int32_t InitParameter() override;

  /*
   * run kernel.
   * @param param: kernel context
   * @return int32_t: 0 indicates success, whilWe the others fail
   */
  int32_t RunKernel(void *param) override;

 private:
  CpuKernelCache(const CpuKernelCache &) = delete;
  CpuKernelCache(CpuKernelCache &&) = delete;
  CpuKernelCache &operator=(const CpuKernelCache &) = delete;
  CpuKernelCache &operator=(CpuKernelCache &&) = delete;

  /*
   * update framework output tensor shape.
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t UpdateFWKOutputShape(
      bool unknown_shape, const CpuKernelContext &ctx,
      std::vector<FWKAdapter::ShapeAndType *> &output_shape_and_type);

  /*
   * get shape information from framework.
   * @param dims: shape information
   */
  void GetDimsFromShapeAndType(const FWKAdapter::ShapeAndType *shape_and_type,
                               std::vector<int64_t> &dims);

  /*
   * update tensor information.
   * @param ctx: kernel context
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t UpdateTensor(
      const std::vector<uint64_t> &io_addrs, bool unknown_shape,
      const std::vector<FWKAdapter::ShapeAndType *> &input_shape_and_type,
      const std::vector<FWKAdapter::ShapeAndType *> &output_shape_and_type,
      CpuKernelContext &ctx);

  /*
   * parse extend tensor shape types information.
   * @param ext_info: extend information
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t ParseExtShapeType(const FWKAdapter::ExtInfo *ext_info,
                             bool &unknown_shape);

  /*
   * parse extend tensor bitmap information.
   * @param ext_info: extend information
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t ParseExtBitMap(const FWKAdapter::ExtInfo *ext_info,
                          bool &unknown_shape);

  /*
   * parse extend tensor shape and types information.
   * @param ext_info: extend information
   * @param shape_and_type: shape and types from extend information
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t ParseExtShapeAndType(
      bool unknown_shape, FWKAdapter::ExtInfo *ext_info,
      std::vector<FWKAdapter::ShapeAndType *> &shape_and_type);

  /*
   * parse extend session information.
   * @param ext_info: extend information
   * @param kernel_id: kernel id from extend information
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t ParseExtSessionInfo(FWKAdapter::ExtInfo *ext_info,
                               uint64_t &kernel_id);

  /*
   * parse extend async wait info
   * @param ext_info : extend infomation
   * @param wait_type: event wait type
   * @param wait_id : event wait id
   * @return uint32_t: 0 indicates success, while the others fail 
   */
  uint32_t ParseAsyncWait(FWKAdapter::ExtInfo *ext_info,
                          uint8_t &wait_type,
                          uint32_t &wait_id);

  /*
   * parse extend information.
   * @param param_head: kernel context
   * @param has_session_info: whether has session info in extend info
   * @param kernel_id: kernel id
   * @param wait_type: event wait type
   * @param wait_id : event wait id
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t ParseExtMsg(
      AicpuParamHead *param_head, bool &has_session_info, uint64_t &kernel_id,
      bool &unknown_shape,
      bool &async_flag,
      uint8_t &wait_type,
      uint32_t &wait_id,
      std::vector<FWKAdapter::ShapeAndType *> &input_shape_and_type,
      std::vector<FWKAdapter::ShapeAndType *> &output_shape_and_type);

  /*
   * parse io address.
   * @param param_head: kernel context
   * @param io_addrs: kernel inputs and outputs adress
   * @param nodedef: kernel node def
   * @param nodedef_len: kernel node def length
   * @return uint32_t: 0 indicates success, while the others fail
   */
  uint32_t ParseIoAddr(AicpuParamHead *param_head,
                       std::vector<uint64_t> &io_addrs, char *&nodedef,
                       uint32_t &nodedef_len);

  /*
   * get cpu kernel context from cache
   * @param has_sess_info: whether has session info
   * @param kernel_id: kernel id, the key of cache
   * @return uint32_t: 0 indicates success, while the others fail
   */
  std::shared_ptr<CpuKernelContext> GetCpuKernelContext(
      bool has_sess_info, uint64_t kernel_id, const char *nodedef,
      uint32_t nodedef_len, std::shared_ptr<NodeDef> &nodedef_proto);

  /*
   * get bit status on pos
   * @param num: input number
   * @param pos: bit pos
   * @return bool: bit is 1 or 0
   */
  bool GetBitStatus(uint64_t num, uint64_t pos);
};
}  // namespace aicpu
#endif  // AICPU_CPU_KERNEL_CACHE_H_
