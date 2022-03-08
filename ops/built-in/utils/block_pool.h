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

#ifndef __BLOCK_POOL_H__
#define __BLOCK_POOL_H__

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <functional>
#include <type_traits>
#include <limits>
#include <mutex>

#include "block_store.h"

namespace memutil {

/**
 * @brief Singletone Memory Poll backed by BlockStore
 *
 */
class BlockPool {
 public:
  static void* malloc(size_t size) {
    auto& p = get_instance();
    return p.malloc_impl(size);
  }

  static void free(void* block) {
    auto& p = get_instance();
    return p.free_impl(block);
  }

 private:
  static BlockPool& get_instance() {
    /* Return a new instance for each thread, so no locking is needed when doing malloc/free  */
    static thread_local BlockPool instance;
    return instance;
  }

  void* malloc_impl(size_t size);
  void free_impl(void* block);
  void* pool_malloc(size_t size);

  void init();
  BlockPool() {
    init();
  };

  /**
   * @brief Block size and count of BlockStore
   */
  struct BlockDesc {
    BlockDesc(size_t s, size_t c) : size(s), count(c){};
    size_t size{0};
    size_t count{0};
  };

  struct object_creator {
    /* This class is only for BlockPool singleton initialization */
    object_creator() {
      BlockPool::get_instance();
    }
  };
  static object_creator create_object;  // Init BlockPool in main thread.
  static const int MAX_STORE = 5;
  static const int BLOCK_BASE_SIZE = 64;
  const uint16_t DEFAULT_TAG = 0x1337;
  const uint16_t SYS_TAG = 0xfeed;
  std::mutex guard_;

  const std::array<BlockDesc, MAX_STORE> STORE_IDX = {
      BlockDesc(BLOCK_BASE_SIZE, 1024), BlockDesc(BLOCK_BASE_SIZE * 4, 256), BlockDesc(BLOCK_BASE_SIZE * 16, 128),
      BlockDesc(BLOCK_BASE_SIZE * 64, 64), BlockDesc(BLOCK_BASE_SIZE * 256, 64)};

  using StoreArray = std::array<BlockStore, MAX_STORE>;
  StoreArray block_store_array_;
  BlockStore* get_store(size_t req_size);
};

}  // namespace memutil

#endif