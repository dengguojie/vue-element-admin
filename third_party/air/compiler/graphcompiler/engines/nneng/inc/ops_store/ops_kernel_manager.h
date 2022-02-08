/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_OPS_STORE_OPS_KERNEL_MANAGER_H_
#define FUSION_ENGINE_INC_OPS_STORE_OPS_KERNEL_MANAGER_H_

#include <string>
#include <map>
#include <mutex>

#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "ops_store/sub_op_info_store.h"

namespace fe {
class OpsKernelManager {
public:
  static OpsKernelManager &Instance(const std::string &engine_name);
  /*
   * initialize the ops kernel manager
   * param[in] the options of init
   * param[in] engineName
   * param[in] socVersion soc version from ge
   * return Status(SUCCESS/FAILED)
   */
  Status Initialize();

  /*
   * to release the source of fusion manager
   * return Status(SUCCESS/FAILED)
   */
  Status Finalize();

  void GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const;

  SubOpInfoStorePtr GetSubOpsKernelByStoreName(const std::string &store_name);

  SubOpInfoStorePtr GetSubOpsKernelByImplType(const OpImplType &op_impl_type);

  OpKernelInfoPtr GetOpKernelInfoByOpType(const std::string &store_name, const std::string &op_type);

  OpKernelInfoPtr GetOpKernelInfoByOpType(const OpImplType &op_impl_type, const std::string &op_type);

  OpKernelInfoPtr GetOpKernelInfoByOpDesc(const ge::OpDescPtr &op_desc_ptr);

  OpKernelInfoPtr GetHighPrioOpKernelInfo(const std::string &op_type);

  Status AddSubOpsKernel(SubOpInfoStorePtr sub_op_info_store_ptr);

private:
  explicit OpsKernelManager(const std::string &engine_name);
  ~OpsKernelManager();
  bool is_init_;
  std::string engine_name_;
  std::map<std::string, SubOpInfoStorePtr> sub_ops_kernel_map_;
  std::map<OpImplType, SubOpInfoStorePtr> sub_ops_store_map_;
  std::mutex ops_kernel_manager_lock_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_OPS_STORE_OPS_KERNEL_MANAGER_H_
