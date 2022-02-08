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

#ifndef FUSION_ENGINE_INC_COMMON_L2_STREAM_INFO_H_
#define FUSION_ENGINE_INC_COMMON_L2_STREAM_INFO_H_

#include <map>
#include <string>
#include <mutex>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "runtime/base.h"
#include "common/l2fusion_struct.h"

namespace fe {
class StreamL2Info {
 public:
  StreamL2Info(const StreamL2Info &) = delete;
  StreamL2Info &operator=(const StreamL2Info &) = delete;
  static StreamL2Info& Instance();
  Status GetStreamL2Info(const rtStream_t &stream_id, std::string node_name, TaskL2Info_t *&l2_data,
                         std::string batch_label);
  Status SetStreamL2Info(const rtStream_t &stream_id, TaskL2InfoFEMap_t &l2_alloc_res, std::string batch_label);

 private:
  StreamL2Info();
  ~StreamL2Info();
  mutable std::mutex stream_l2_mutex_;
  std::map<std::string, TaskL2InfoFEMap_t> stream_l2_map_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_INC_COMMON_L2_STREAM_INFO_H_
