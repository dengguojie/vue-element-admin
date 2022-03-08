#include <gtest/gtest.h>
#include <string>
#include <list>
#include <array>
#include <vector>
#include <random>
#include <map>
#include <utility>
#include <thread>

#include "block_store.h"
#include "block_pool.h"
#include "pool_allocator.h"

using namespace memutil;
using namespace std;

class MemPoolTestUtil {
 public:
  template<int N>
  static int randint() {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_int_distribution<> distrib(1, N);
    return distrib(gen);
  }

  using data_map = map<void*, long>;
  static int verify_data(const data_map &data) {
    int incorrect = 0;
    for (const auto &elem : data) {
      if (*static_cast<long*>(elem.first) != elem.second) {
        incorrect++;
      }
    }
    return incorrect;
  }

  static void free_random_block(data_map &data, BlockStore &bs) {
    int c = data.size();
    if (c == 0) {
      return;
    }
    int rand = randint<9999999>() % c;
    for (auto itr = data.begin(); itr != data.end(); itr++) {
      if (rand-- != 0) {
        continue;
      }
      bs.free(itr->first);
      data.erase(itr);
      break;
    }
  }
};

class BlockStoreTest: public ::testing::Test, public MemPoolTestUtil {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(BlockStoreTest, init1) {
  BlockStore bs;
  int rc = bs.init(111, 0, 100);
  EXPECT_NE(rc, 0);

  rc = bs.init(111, 10, 0);
  EXPECT_NE(rc, 0);

  rc = bs.init(111, 128, 100);
  EXPECT_EQ(rc, 0);
}

TEST_F(BlockStoreTest, init2) {
  BlockStore bs;
  int rc = bs.init(111, 1, 100);
  EXPECT_EQ(rc, 0);

  EXPECT_EQ(bs.get_block_size(), 64);
  EXPECT_EQ(bs.get_tag(), 111);

  {
    BlockStore bs;
    int rc = bs.init(111, 63, 100);
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(bs.get_block_size(), 64);
  }

  {
    BlockStore bs;
    int rc = bs.init(111, 64, 100);
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(bs.get_block_size(), 64);
  }

  {
    BlockStore bs;
    int rc = bs.init(111, 65, 100);
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(bs.get_block_size(), 128);
  }

  {
    BlockStore bs;
    int rc = bs.init(111, 127, 100);
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(bs.get_block_size(), 128);
  }
}

TEST_F(BlockStoreTest, uninit1) {
  BlockStore bs;
  int rc = bs.init(111, 1, 100);
  EXPECT_EQ(rc, 0);

  void *p = bs.alloc();
  EXPECT_NE(p, nullptr);
  bs.uninit();

  rc = bs.init(111, 128, 100);
  EXPECT_EQ(rc, 0);
}

TEST_F(BlockStoreTest, alloc1) {
  BlockStore bs;
  int count = 10;
  int rc = bs.init(111, 1, count);
  EXPECT_EQ(rc, 0);
  for (int i = 0; i < count; i++) {
    void *p = bs.alloc();
    EXPECT_NE(p, nullptr);
  }
  void *p = bs.alloc();
  EXPECT_EQ(p, nullptr);
}

TEST_F(BlockStoreTest, free1) {
  BlockStore bs;
  constexpr int count = 10;
  array<void*, count> ptr;
  int rc = bs.init(111, 1, count);
  EXPECT_EQ(rc, 0);
  for (int i = 0; i < count; i++) {
    ptr[i] = bs.alloc();
    EXPECT_NE(ptr[i], nullptr);
  }

  for (int i = 0; i < count; i++) {
    bs.free(ptr[i]);
  }

  for (int i = 0; i < count; i++) {
    ptr[i] = bs.alloc();
    EXPECT_NE(ptr[i], nullptr);
  }
  void *p = bs.alloc();
  EXPECT_EQ(p, nullptr);
}

TEST_F(BlockStoreTest, mix_alloc_free) {
  BlockStore bs;
  constexpr int count = 100;
  int rc = bs.init(111, 22, count);
  EXPECT_EQ(rc, 0);

  int repeat = 10000;
  data_map curr_allocated;

  while (repeat--) {
    int what_to_do = randint<100>();
    if (what_to_do < 50) {
      free_random_block(curr_allocated, bs);
      int nwrong = verify_data(curr_allocated);
      EXPECT_EQ(nwrong, 0);
      continue;
    }

    void *p = bs.alloc();
    if (curr_allocated.size() >= count) {
      EXPECT_EQ(p, nullptr);
      continue;
    }

    EXPECT_NE(p, nullptr);
    long data = randint<999999>();
    *static_cast<long*>(p) = data;
    int nwrong = verify_data(curr_allocated);
    EXPECT_EQ(nwrong, 0);
    curr_allocated.emplace(p, data);
  }
}


class BlockPoolTest: public ::testing::Test, public MemPoolTestUtil {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};


TEST_F(BlockPoolTest, malloc1) {
  for (int i = 0; i < 10000; i++) {
    long *p = static_cast<long*>(BlockPool::malloc(100));
    EXPECT_NE(p, nullptr);
    BlockPool::free(p);
  }
}

TEST_F(BlockPoolTest, malloc2) {
  set<long*> ptr;
  int repeat = 10000;
  for (int i = 0; i < repeat; i++) {
    long *p = static_cast<long*>(BlockPool::malloc(64));
    memset(p, 9, 64);
    EXPECT_NE(p, nullptr);
    ptr.emplace(p);
  }

  for (auto &elem : ptr) {
    for (int i = 0; i < 64; i++) {
      uint8_t t = *reinterpret_cast<uint8_t*>(elem);
      EXPECT_EQ(t, 9);
    }
    BlockPool::free(elem);
  }
}

void thrd_run() {
  int count = 1000;
  long *ptr [1000];
  while (count--) {
    for (int i = 0; i < 1000; i++) {
      long *p = static_cast<long*>(BlockPool::malloc(64));
      *p = i+100;
      ptr[i] = p;
    }

    for (int i = 0; i < 1000; i++) {
      EXPECT_EQ(*ptr[i], i+100);
      BlockPool::free(ptr[i]);
    }
  }
}

TEST_F(BlockPoolTest, malloc3) {
  thrd_run();
}


TEST_F(BlockPoolTest, thrd_malloc) {
  vector<thread> thrds;
  for (int i = 0; i < 10; i++) {
    thrds.emplace_back(move(thread(thrd_run)));
  }
  for (auto &thrd : thrds) {
    thrd.join();
  }
}

TEST_F(BlockPoolTest, sys_malloc) {
  int count = 1000;
  long *ptr[10000];
  while (count--) {
    for (int i = 0; i < 1000; i++) {
      long *p = static_cast<long*>(malloc(64));
      *p = 100 + i;
      ptr[i] = p;
    }
    for (int i = 0; i < 1000; i++) {
      EXPECT_EQ(*ptr[i], i+100);
      BlockPool::free(ptr[i]);
    }
  }
}

class PoolAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(PoolAllocatorTest, vector1) {
  vector<int, PoolAllocator<int>> vec;
  int repeat = 10000;
  for (int i = 0; i < repeat; i++) {
    vec.emplace_back(i);
  }
  EXPECT_EQ(vec.size(), repeat);
  for (int i = 0; i < repeat; i++) {
    EXPECT_EQ(vec[i], i);
  }
  vec.clear();
  EXPECT_EQ(vec.size(), 0);
}