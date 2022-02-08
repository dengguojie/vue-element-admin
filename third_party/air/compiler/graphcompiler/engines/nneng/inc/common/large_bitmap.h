/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_LARGE_BITMAP_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_LARGE_BITMAP_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/fe_log.h"

namespace fe {
class LargeBitMap {
 public:
  explicit LargeBitMap(size_t size);

  ~LargeBitMap() = default;

  bool operator==(const LargeBitMap &another_bm) const;

  bool operator!=(const LargeBitMap &another_bm) const;

  // set all vector to specific value
  void SetValues(uint64_t value);

  // Get the value on position index
  bool GetBit(size_t index) const;

  // Set the value on position index to 1
  void SetBit(size_t index);

  // Combine two bitmap with rule:
  // If one bit of either one of the two bitmaps is 1,
  // the result of final bitmap is 1.
  void Or(const LargeBitMap &another_bm);
 private:
  // Number of element in vector bits
  size_t size_;

  std::vector<uint64_t> bits_;
};
}
#endif //AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_LARGE_BITMAP_H_
