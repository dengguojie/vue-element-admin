#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include <algorithm>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

namespace {
  template <typename T>
  struct ValueIndex {
    T value;
    int32_t index;
  };

  template <typename T>
  bool CompareDescending(const ValueIndex<T> &one, const ValueIndex<T> &another) {
    if (one.value == another.value) {
      return one.index < another.index;
    }
    return one.value > another.value;
  }
	
  template <typename T>
  bool CompareAscending(const ValueIndex<T> &one, const ValueIndex<T> &another) {
    if (one.value == another.value) {
  	return one.index < another.index;
    }
    return one.value < another.value;
  }
}

class TEST_TOPK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "TopK", "TopK")                               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Input({"k", data_types[1], shapes[1], datas[1]})                        \
      .Output({"values", data_types[2], shapes[2], datas[2]})                  \
      .Output({"indices", data_types[3], shapes[3], datas[3]})                 \
      .Attr("sorted", true)                                                    \
      .Attr("largest", true)                                                   \
      .Attr("dim", -1);

#define CREATE_NODEDEF2(shapes, data_types, datas)                             \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "TopK", "TopK")                               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Input({"k", data_types[1], shapes[1], datas[1]})                        \
      .Output({"values", data_types[2], shapes[2], datas[2]})                  \
      .Output({"indices", data_types[3], shapes[3], datas[3]})                 \
      .Attr("sorted", true)                                                    \
      .Attr("largest", false)                                                  \
      .Attr("dim", -1);
  
#define CREATE_NODEDEF3(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "TopK", "TopK")                               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Input({"k", data_types[1], shapes[1], datas[1]})                        \
      .Output({"values", data_types[2], shapes[2], datas[2]})                  \
      .Output({"indices", data_types[3], shapes[3], datas[3]})                 \
      .Attr("sorted", true)                                                    \
      .Attr("largest", true)                                                   \
      .Attr("dim", -2);

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_TOPK_UT, TestTopK_##aicpu_type##_LARGEST) {                      \
    vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type, DT_INT32};\
    vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}};                     \
    base_type input[24];                                                       \
    SetRandomValue<base_type>(input, 24);                                      \
    vector<ValueIndex<base_type>> output_expect(24);                           \
    for (int i = 0; i < 24; i++) {                                             \
      output_expect[i].index = i;                                              \
      output_expect[i].value = input[i];                                       \
    }                                                                          \
    sort(output_expect.begin(), output_expect.end(),                           \
         CompareDescending<base_type>);                                        \
    int32_t k = 7;                                                             \
    base_type output_value[7] = {(base_type)0};                                \
    int32_t output_index[7] = {0};                                             \
    vector<void *> datas = {(void *)input, (void *)&k, (void *)output_value,   \
                            (void *)output_index};                             \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    for (int i = 0; i < 7; i++) {                                              \
      EXPECT_EQ(output_value[i], output_expect[i].value);                      \
      EXPECT_EQ(output_index[i], output_expect[i].index);                      \
    }                                                                          \
  }                                                                            \
  TEST_F(TEST_TOPK_UT, TestTopK_##aicpu_type##SMALLEST) {                      \
    vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type, DT_INT32};\
    vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}};                     \
    base_type input[24];                                                       \
    SetRandomValue<base_type>(input, 24);                                      \
    vector<ValueIndex<base_type>> output_expect(24);                           \
    for (int i = 0; i < 24; i++) {                                             \
      output_expect[i].index = i;                                              \
      output_expect[i].value = input[i];                                       \
    }                                                                          \
    sort(output_expect.begin(), output_expect.end(),                           \
         CompareAscending<base_type>);                                         \
    int32_t k = 7;                                                             \
    base_type output_value[7] = {(base_type)0};                                \
    int32_t output_index[7] = {0};                                             \
    vector<void *> datas = {(void *)input, (void *)&k, (void *)output_value,   \
                            (void *)output_index};                             \
    CREATE_NODEDEF2(shapes, data_types, datas);                                \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    for (int i = 0; i < 7; i++) {                                              \
      EXPECT_EQ(output_value[i], output_expect[i].value);                      \
      EXPECT_EQ(output_index[i], output_expect[i].index);                      \
    }                                                                          \  
  }                                                                            \
  TEST_F(TEST_TOPK_UT, TestTopK_##aicpu_type##_SECOND_LAST_DIM) {              \
    vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type, DT_INT32};\
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {}, {2, 2, 4}, {2, 2, 4}};    \
    base_type input[24];                                                       \
    for (int i = 0; i < 24; i++) {                                             \
      input[i] = base_type(i + 1);                                             \
    }                                                                          \
    base_type output_value_expect[16] =                                        \
    {base_type(9), base_type(10), base_type(11), base_type(12),                \
     base_type(5), base_type(6), base_type(7), base_type(8),                   \
     base_type(21), base_type(22), base_type(23), base_type(24),               \
     base_type(17), base_type(18), base_type(19), base_type(20)};              \
    int32_t output_index_expect[16] =                                          \
    {2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1};                          \
    int32_t k = 2;                                                             \
    base_type output_value[16] = {(base_type)0};                               \
    int32_t output_index[16] = {0};                                            \
    vector<void *> datas = {(void *)input, (void *)&k, (void *)output_value,   \
                            (void *)output_index};                             \
    CREATE_NODEDEF3(shapes, data_types, datas);                                \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    for (int i = 0; i < 16; i++) {                                             \
      EXPECT_EQ(output_value[i], output_value_expect[i]);                      \
      EXPECT_EQ(output_index[i], output_index_expect[i]);                      \
    }                                                                          \
  } 

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(int8_t, DT_INT8)

ADD_CASE(int16_t, DT_INT16)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint8_t, DT_UINT8)

ADD_CASE(uint16_t, DT_UINT16)

ADD_CASE(uint32_t, DT_UINT32)

ADD_CASE(uint64_t, DT_UINT64)
