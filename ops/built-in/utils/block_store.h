
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef __BLOCK_STORE_H__
#define __BLOCK_STORE_H__

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <functional>
#include <type_traits>
#include <limits>

namespace memutil {

/**
 * @brief Simple memory block managmennt class.
 */
class BlockStore {
 public:
  BlockStore() = default;
  ~BlockStore();
  BlockStore(const BlockStore&) = delete;
  BlockStore& operator=(const BlockStore&) = delete;
  BlockStore(BlockStore&& src);
  BlockStore& operator=(BlockStore&& src);

  int init(uint16_t tag, size_t block_size, size_t block_count);
  void uninit();

  void* alloc();
  void free(void* block);
  size_t get_block_size() const;
  uint16_t get_tag() const;

  static uint16_t get_block_tag(void* block);

  /**
   * @brief MAGIC HEADER for each managed memory block
   *
   */
  static constexpr uint32_t MAGIC = 0x12345678;

  /**
   * @brief User can set a TAG for this BlockStore
   *
   */
  static constexpr uint16_t DEFAULT_TAG = 0;
  using BlockIdx = int32_t;

  enum class BlockState : uint16_t { UNINIT = 0, FREE = 1, ALLOCATED = 2 };

  /**
   * @brief Each block has a header
   */
  struct BlockHeader {
    uint32_t magic{MAGIC};   // magic header
    BlockIdx block_idx{-1};  // block index
    BlockIdx next{-1};       // next free block index
    BlockState block_state{BlockState::UNINIT};
    uint16_t user_tag{DEFAULT_TAG};
  };

 private:
  static const size_t MIN_BLOCK_SIZE = 64;
  static size_t aligned_block_size(size_t req_size) {
    return (req_size + MIN_BLOCK_SIZE - 1) / MIN_BLOCK_SIZE * MIN_BLOCK_SIZE;
  };
  BlockHeader* get_header(int idx) const;

  uint16_t tag_{DEFAULT_TAG};
  size_t block_size_{0};
  size_t block_count_{0};
  void* mem_{nullptr};
  BlockIdx free_head_{-1};  // index of free list head.
};
}  // namespace memutil

#endif
