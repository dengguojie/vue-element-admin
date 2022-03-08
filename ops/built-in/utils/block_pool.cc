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

#include "block_pool.h"

namespace memutil {

/**
 * @brief Initialize each BlockStore
 */
void BlockPool::init() {
  bool succ = false;
  uint16_t tag = DEFAULT_TAG;
  for (size_t i = 0; i < STORE_IDX.size(); i++) {
    int rc = block_store_array_[i].init(tag + i, STORE_IDX[i].size, STORE_IDX[i].count);
    if (rc != 0) {
      break;
    }
  }
  if (succ) {
    return;
  }

  for (size_t i = 0; i < STORE_IDX.size(); i++) {
    block_store_array_[i].uninit();
  }
}

/**
 * @brief Get BlockStore pointer by requested memory size
 *
 * @param req_size memory size
 * @return BlockStore* BlockStore pointer,  return nullptr if no BlockStore found for the requested memroy size
 */
BlockStore* BlockPool::get_store(size_t req_size) {
  if (req_size == 0 || req_size > STORE_IDX[STORE_IDX.size() - 1].size) {
    return nullptr;
  }

  int idx = 0;
  int log4 = (req_size + BLOCK_BASE_SIZE - 1) / BLOCK_BASE_SIZE;
  while (log4 > 1) {
    log4 /= 4;
    idx++;
  }

  return &block_store_array_[idx];
}

void* BlockPool::malloc_impl(size_t size) {
  void* p = pool_malloc(size);
  if (p != nullptr) {
    return p;
  }

  p = std::malloc(sizeof(BlockStore::BlockHeader) + size);
  if (p == nullptr) {
    return nullptr;
  }

  BlockStore::BlockHeader* head = static_cast<BlockStore::BlockHeader*>(p);
  head->user_tag = SYS_TAG;
  return head + 1;
}

void* BlockPool::pool_malloc(size_t size) {
  const std::lock_guard<std::mutex> guard(guard_);
  BlockStore* store = get_store(size);
  if (store == nullptr) {
    return nullptr;
  }
  return store->alloc();
}

void BlockPool::free_impl(void* block) {
  uint16_t tag = BlockStore::get_block_tag(block);
  if (tag == SYS_TAG) {
    std::free(static_cast<BlockStore::BlockHeader*>(block) - 1);
    return;
  }

  size_t idx = tag - DEFAULT_TAG;
  if (idx > STORE_IDX.size()) {
    // FATAL: try to free block not belong to this BlockPool
    return;
  }

  BlockStore& store = block_store_array_[idx];
  const std::lock_guard<std::mutex> guard(guard_);
  store.free(block);
}

BlockPool::object_creator BlockPool::create_object;

}  // namespace memutil