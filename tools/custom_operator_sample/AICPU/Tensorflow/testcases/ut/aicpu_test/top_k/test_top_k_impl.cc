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

class TopKTest : public testing::Test {};

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

TEST_F(TopKTest, TestTopK_1) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}};                     
    float input[24];                                                       
    SetRandomValue<float>(input, 24);                                      
    vector<ValueIndex<float>> output_expect(24);                           
    for (int i = 0; i < 24; i++) {                                             
      output_expect[i].index = i;                                              
      output_expect[i].value = input[i];                                       
    }                                                                          
    sort(output_expect.begin(), output_expect.end(),                           
         CompareAscending<float>);                                         
    int32_t k = 7;                                                             
    float output_value[7] = {(float)0};                                
    int32_t output_index[7] = {0};                                             
    vector<void *> datas = {(void *)input, (void *)&k, (void *)output_value,   
                            (void *)output_index};                             
    CREATE_NODEDEF2(shapes, data_types, datas);                                
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              
    for (int i = 0; i < 7; i++) {                                              
      EXPECT_EQ(output_value[i], output_expect[i].value);                      
      EXPECT_EQ(output_index[i], output_expect[i].index);                      
    }                                                                          
}  











/*

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
*/