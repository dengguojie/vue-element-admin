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

#ifndef FUSION_ENGINE_INC_OPS_STORE_SUB_OP_INFO_STORE_H_
#define FUSION_ENGINE_INC_OPS_STORE_SUB_OP_INFO_STORE_H_

#include <string>
#include <vector>
#include <map>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "ops_store/op_kernel_info.h"
#include "ops_store/op_kernel_info_constructor.h"

namespace fe {
class SubOpInfoStore;
using SubOpInfoStorePtr = std::shared_ptr<SubOpInfoStore>;

class SubOpInfoStore {
 public:
  SubOpInfoStore(const FEOpsStoreInfo &ops_store_info);
  virtual ~SubOpInfoStore();

  /*
   * @ingroup fe
   * @brief : Initialize the SubOpInfoStore, load the op info from the specific.json file
   * @param[in] options: none
   * @return : SUCCESS/FAILED
   */
  Status Initialize(const std::string &engine_name);

  /*
   * @ingroup fe
   * @brief : finalize the SubOpInfoStore, clear all op info and op kernel info;
   * @param[in] None
   * @return : SUCCESS/FAILED
   */
  Status Finalize();

  const std::map<std::string, OpKernelInfoPtr>& GetAllOpKernels();

  OpKernelInfoPtr GetOpKernelByOpType(const std::string &op_type);

  Status GetOpContentByOpType(const std::string &op_type, OpContent &op_content);

  Status SetOpContent(const OpContent &op_content);

  Status LoadOpJsonFile(const std::string &json_file_path);

  Status ConstructOpKernelInfo(const std::string &engine_name);

  const std::string& GetOpsStoreName() const;

  const OpImplType& GetOpImplType() const;

  Status LoadModifyMixlistJson(const std::string &modify_mixlist_path);

  Status LoadCustomJsonFile(const std::string &modify_mixlist_path);

  Status UpdateOpInfoStore(const std::map<std::string, uint8_t> &update_map);

  void UpdateStrToOpContent(OpContent &op_content, const std::string key1, const std::string key2, std::string &value);

 private:
  bool init_flag_;
  FEOpsStoreInfo ops_store_info_;
  /* std::string represents op_type */
  std::map<std::string, OpKernelInfoPtr> op_kernel_info_map_;
  std::map<std::string, OpContent> op_content_map_;

  /*
   * @brief : read the op information json file, and load the op
   * information to this sub store;
   */
  Status LoadOpInfo(const std::string &filename);


};

}  // namespace fe

#endif  // FUSION_ENGINE_INC_OPS_STORE_SUB_OP_INFO_STORE_H_