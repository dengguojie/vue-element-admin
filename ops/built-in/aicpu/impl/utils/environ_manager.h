/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef AICPU_UTILS_ENVIRON_MANAGER_H
#define AICPU_UTILS_ENVIRON_MANAGER_H

#include <utility>
#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include "utils/environ.h"
#include "utils/kernel_util.h"

namespace aicpu {
class EnvironMgr {
 public:
  static EnvironMgr &GetInstance() noexcept {
    static EnvironMgr instance;
    return instance;
  }

  EnvironMgr(const EnvironMgr &) = delete;
  EnvironMgr(EnvironMgr &&) = delete;
  EnvironMgr &operator=(const EnvironMgr &) = delete;
  EnvironMgr &operator=(EnvironMgr &&) = delete;

  // Create the env object and return the unique env handle.
  int64_t Create();

  EnvironPtr Get(int64_t handle);

  void Clear();

  // Check whether the inputs of EnvironGet kernel or EnvironSet kernel are valid.
  bool CheckEnvInput(const CpuKernelContext &ctx);
  // Check whether is scalar tensor. Environ handle and env key only support scalar tensor currently.
  bool IsScalarTensor(std::vector<int64_t> shape);

 private:
  EnvironMgr() = default;
  ~EnvironMgr() = default;

  // Store the envs in map, as <handle, env>.
  std::map<int64_t, EnvironPtr> envs_;

  int64_t env_handles_count_{0};

  std::mutex mutex;
};
}  // namespace aicpu

#endif  // AICPU_UTILS_ENVIRON_MANAGER_H
