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

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <functional>
#include <type_traits>
#include <limits>

#include "block_store.h"

namespace memutil {

/**
 * @brief Initialize the BlockStore
 *
 * @param tag User-defined tag
 * @param block_size size of each block
 * @param block_count count of blocks
 * @return int 0 for success, others for fail
 */
int BlockStore::init(uint16_t tag, size_t block_size, size_t block_count) {
  if (block_size == 0 || block_count == 0) {
    return -1;
  }
  tag_ = tag;
  block_size_ = aligned_block_size(block_size);
  block_count_ = block_count;
  if (block_count > static_cast<size_t>(std::numeric_limits<BlockIdx>::max())) {
    return -1;
  }

  mem_ = std::malloc((block_size_ + sizeof(BlockHeader)) * block_count_);
  if (mem_ == nullptr) {
    return -1;
  }

  for (BlockIdx i = 0; i < static_cast<BlockIdx>(block_count_); i++) {
    BlockHeader* head = get_header(i);
    new (head) BlockHeader;
    head->user_tag = tag;
    head->block_idx = i;
    if (i < static_cast<BlockIdx>(block_count_) - 1) {
      head->next = i + 1;
    } else {
      head->next = -1;
    }
  }
  free_head_ = 0;
  return 0;
}

void BlockStore::uninit() {
  if (mem_) {
    std::free(mem_);
    mem_ = nullptr;
  }

  tag_ = DEFAULT_TAG;
  block_size_ = 0;
  block_count_ = 0;
  free_head_ = -1;
}

BlockStore::~BlockStore() {
  if (mem_) {
    std::free(mem_);
  }
}

void* BlockStore::alloc() {
  if (free_head_ == -1) {
    return nullptr;
  }

  BlockHeader* head = get_header(free_head_);
  free_head_ = head->next;
  head->block_state = BlockState::ALLOCATED;
  return head + 1;
}

void BlockStore::free(void* block) {
  if (block == nullptr) {
    return;
  }

  BlockHeader* head = static_cast<BlockHeader*>(block) - 1;
  if (head->magic != BlockStore::MAGIC || head->user_tag != tag_) {
    // FATAL, try to free memory no belong to this BlockStore
    return;
  }
  BlockIdx tmp = free_head_;
  free_head_ = head->block_idx;
  head->next = tmp;
  head->block_state = BlockState::FREE;
}

BlockStore::BlockHeader* BlockStore::get_header(int idx) const {
  if (idx < 0 || static_cast<size_t>(idx) >= block_count_) {
    return nullptr;
  }
  uint8_t* p = static_cast<uint8_t*>(mem_);
  p += idx * (block_size_ + sizeof(BlockHeader));
  return static_cast<BlockHeader*>(static_cast<void*>(p));
}

size_t BlockStore::get_block_size() const {
  return block_size_;
}

uint16_t BlockStore::get_tag() const {
  return tag_;
}

uint16_t BlockStore::get_block_tag(void* block) {
  BlockHeader* head = static_cast<BlockHeader*>(block) - 1;
  return head->user_tag;
}

}  // namespace memutil