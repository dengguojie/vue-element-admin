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

#ifndef DOMI_OP_ARG_OP_H_
#define DOMI_OP_ARG_OP_H_
#include "common/op_def/operator.h"

namespace ge {
class ArgOpOperator : public domi::Operator {
 public:
  ArgOpOperator();

  ~ArgOpOperator();

  ArgOpOperator &Name(const std::string &name);

  ArgOpOperator &Index(int64_t index);

  int64_t GetIndex() const;
};
}  // namespace ge

#endif  // DOMI_OP_ARG_OP_H_