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

#ifndef __POOL_ALLOCATOR_H__
#define __POOL_ALLOCATOR_H__

#include <cstdlib>
#include <new>
#include <limits>

#include "block_pool.h"

namespace memutil {

template <class T>
struct PoolAllocator {
  using value_type = T;
  using size_t = std::size_t;

  PoolAllocator() = default;

  template <class U>
  constexpr PoolAllocator(const PoolAllocator<U>&) noexcept {
  }

  T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    T* p = static_cast<T*>(BlockPool::malloc(n * sizeof(T)));
    if (p != nullptr) {
      return p;
    }

    throw std::bad_alloc();
  }

  void deallocate(T* p, size_t n) noexcept {
    BlockPool::free(p);
  }
};

template <class T, class U>
bool operator==(const PoolAllocator<T>&, const PoolAllocator<U>&) {
  return true;
}

template <class T, class U>
bool operator!=(const PoolAllocator<T>&, const PoolAllocator<U>&) {
  return false;
}

}  // namespace memutil

#endif