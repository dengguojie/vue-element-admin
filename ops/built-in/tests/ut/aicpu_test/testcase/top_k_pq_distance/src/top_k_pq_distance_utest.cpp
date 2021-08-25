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

class TEST_TOPK_PQ_DISTANCE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, k,group_size)                       \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(), "TopKPQDistance", "TopKPQDistance")       \
      .Input({"actual_count0", data_types[3], shapes[3], datas[0]})          \
      .Input({"actual_count1", data_types[3], shapes[3], datas[1]})          \
      .Input({"pq_distance0", data_types[1], shapes[0], datas[2]})          \
      .Input({"pq_distance1", data_types[1], shapes[0], datas[3]})          \
      .Input({"grouped_extreme_distance0", data_types[1], shapes[2], datas[4]}) \
      .Input({"grouped_extreme_distance1", data_types[1], shapes[2], datas[5]}) \
      .Input({"pq_ivf0", data_types[2], shapes[0], datas[6]})               \
      .Input({"pq_ivf1", data_types[2], shapes[0], datas[7]})               \
      .Input({"pq_index0", data_types[2], shapes[0], datas[8]})             \
      .Input({"pq_index1", data_types[2], shapes[0], datas[9]})             \
      .Output({"topk_distance", data_types[1], shapes[1], datas[10]})       \
      .Output({"topk_ivf", data_types[2], shapes[1], datas[11]})            \
      .Output({"topk_index", data_types[2], shapes[1], datas[12]})          \
      .Attr("order", "DES")                                                 \
      .Attr("k", k)                                                         \
      .Attr("group_size", group_size)

#define CREATE_NODEDEF2(shapes, data_types, datas, k,group_size)                       \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(), "TopKPQDistance", "TopKPQDistance")       \
      .Input({"actual_count0", data_types[3], shapes[3], datas[0]})          \
      .Input({"pq_distance0", data_types[1], shapes[0], datas[1]})          \
      .Input({"grouped_extreme_distance0", data_types[1], shapes[2], datas[2]}) \
      .Input({"pq_ivf0", data_types[2], shapes[0], datas[3]})               \
      .Input({"pq_index0", data_types[2], shapes[0], datas[4]})             \
      .Output({"topk_distance", data_types[1], shapes[1], datas[5]})       \
      .Output({"topk_ivf", data_types[2], shapes[1], datas[6]})            \
      .Output({"topk_index", data_types[2], shapes[1], datas[7]})          \
      .Attr("order", "DES")                                                 \
      .Attr("k", k)                                                         \
      .Attr("group_size", group_size)


TEST_F(TEST_TOPK_PQ_DISTANCE_UT, DATA_TYPE_FLOAT) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_INT32};
  std::string order = "DES";
  constexpr int32_t k = 5;
  constexpr int32_t group_size = 2;  
  constexpr int32_t actual_count = 6;

  vector<vector<int64_t>> shapes = {{actual_count}, {k}, {actual_count/group_size}, {}};
  float pq_distance0[actual_count] = {1,2,3,4,12,13}; 
  float grouped_extreme_distance0[actual_count/group_size] = {2,4,13};
  int32_t pq_ivf0[actual_count] = {1,1,1,1,1,1};
  int32_t pq_index0[actual_count] = {1,2,3,4,5,6};


  float pq_distance1[actual_count] = {5, 1, 4, 2, 3, 2};
  float grouped_extreme_distance1[actual_count/group_size] = {5,4,3};
  int32_t pq_ivf1[actual_count] = {1,1,1,1,1,1};
  int32_t pq_index1[actual_count] = {7,8,9,10,11,12};

  // ouput
  float topk_distance[k];
  int32_t topk_ivf[k];
  int32_t topk_index[k];
  vector<void*> datas = {(void*)(&actual_count),
                         (void*)(&actual_count),
                         (void*)pq_distance0,
                         (void*)pq_distance1,
                         (void*)grouped_extreme_distance0,
                         (void*)grouped_extreme_distance1,
                         (void*)pq_ivf0,
                         (void*)pq_ivf1,
                         (void*)pq_index0,
                         (void*)pq_index1,
                         (void*)topk_distance,
                         (void*)topk_ivf,
                         (void*)topk_index};
  CREATE_NODEDEF(shapes, data_types, datas, k, group_size);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<float> expect_vec(pq_distance0, pq_distance0 + actual_count);
  expect_vec.insert(expect_vec.end(), pq_distance1, pq_distance1 + actual_count);
  sort(expect_vec.begin(),expect_vec.end(),[order](float a, float b) { 
    if(order == "DES"){
      return a > b; 
    }
    return a < b;
  });
  for (int i = 0; i < k; i++) {
    EXPECT_EQ(expect_vec[i], topk_distance[i]);
  }
}

TEST_F(TEST_TOPK_PQ_DISTANCE_UT, DATA_TYPE_FLOAT_5INPUT) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_INT32};
  std::string order = "DES";
  constexpr int32_t k = 5;
  constexpr int32_t group_size = 2;  
  constexpr int32_t actual_count = 12;

  vector<vector<int64_t>> shapes = {{actual_count}, {k}, {actual_count/group_size}, {}};
  float pq_distance0[actual_count] = {1,2,3,4,12,13,5, 1, 4, 2, 3, 2}; 
  float grouped_extreme_distance0[actual_count/group_size] = {2,4,13,5,4,3};
  int32_t pq_ivf0[actual_count] = {1,1,1,1,1,1,1,1,1,1,1,1};
  int32_t pq_index0[actual_count] = {1,2,3,4,5,6,7,8,9,10,11,12};

  // ouput
  float topk_distance[k];
  int32_t topk_ivf[k];
  int32_t topk_index[k];
  vector<void*> datas = {(void*)(&actual_count),
                         (void*)pq_distance0,
                         (void*)grouped_extreme_distance0,
                         (void*)pq_ivf0,
                         (void*)pq_index0,
                         (void*)topk_distance,
                         (void*)topk_ivf,
                         (void*)topk_index};
  CREATE_NODEDEF2(shapes, data_types, datas, k, group_size);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<float> expect_vec(pq_distance0, pq_distance0 + actual_count);
  sort(expect_vec.begin(),expect_vec.end(),[order](float a, float b) { 
    if(order == "DES"){
      return a > b; 
    }
    return a < b;
  });
  for (int i = 0; i < k; i++) {
    EXPECT_EQ(expect_vec[i], topk_distance[i]);
  }
}

template <typename T>
void fillExtreme(T extremeArr[], T pq_distance[], int actual_count, int group_size, string &order) {
  bool is_max = order == "DES" ? true : false;
  int32_t grp_num = actual_count / group_size;
  for (int32_t i = 0; i < grp_num; i++) {
    T temp_extreme = pq_distance[i * group_size];
    for (int32_t j = i * group_size; j < (i + 1) * group_size; j++) {
      if (is_max) {
        if (pq_distance[j] > temp_extreme) {
          temp_extreme = pq_distance[j];
        }
      } else {
        if (pq_distance[j] < temp_extreme) {
          temp_extreme = pq_distance[j];
        }
      }
    }
    extremeArr[i] = temp_extreme;
  }
}

TEST_F(TEST_TOPK_PQ_DISTANCE_UT, DATA_TYPE_FLOAT_TIME) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_INT32};
  std::string order = "DES";
  constexpr int32_t k = 1024;
  constexpr int32_t group_size = 60;  
  constexpr int32_t actual_count0 = 120000;
  constexpr int32_t actual_count1 = 120000; 

  vector<float> expect_vec;

  vector<vector<int64_t>> shapes = {{actual_count0}, {k}, {actual_count0/group_size}, {}, {actual_count1}, {actual_count1/group_size}};
  float pq_distance0[actual_count0];
  float grouped_extreme_distance0[actual_count0/group_size];
  int32_t pq_ivf0[actual_count0];
  int32_t pq_index0[actual_count0];

  for (int32_t i = 0; i < actual_count0; i++) {
    pq_distance0[i] = (rand() % 200 + 1);
    pq_ivf0[i] = i;
    pq_index0[i] = 1;
    expect_vec.emplace_back(pq_distance0[i]);
  }
  fillExtreme(grouped_extreme_distance0, pq_distance0, actual_count0, group_size, order);

  float pq_distance1[actual_count1];
  float grouped_extreme_distance1[actual_count1/group_size];
  int32_t pq_ivf1[actual_count1];
  int32_t pq_index1[actual_count1];

  for (int32_t i = 0; i < actual_count1; i++) {
    pq_distance1[i] = (rand() % 200000 + 1);
    pq_ivf1[i] = i;
    pq_index1[i] = 1;
    expect_vec.emplace_back(pq_distance1[i]);
  }
  fillExtreme(grouped_extreme_distance1, pq_distance1, actual_count1, group_size, order);

  // ouput
  float topk_distance[k];
  int32_t topk_ivf[k];
  int32_t topk_index[k];
  vector<void*> datas = {(void*)(&actual_count0),
                         (void*)(&actual_count1),
                         (void*)pq_distance0,
                         (void*)pq_distance1,
                         (void*)grouped_extreme_distance0,
                         (void*)grouped_extreme_distance1,
                         (void*)pq_ivf0,
                         (void*)pq_ivf1,
                         (void*)pq_index0,
                         (void*)pq_index1,
                         (void*)topk_distance,
                         (void*)topk_ivf,
                         (void*)topk_index};
  CREATE_NODEDEF(shapes, data_types, datas, k, group_size);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  sort(expect_vec.begin(), expect_vec.end(), [order](float a, float b) {
    if (order == "DES") {
      return a > b;
    } else { 
      return a < b;
    } 
  });
 
  for (int i = 0; i < k; i++) {
    EXPECT_EQ(expect_vec[i], topk_distance[i]);
  } 
} 

TEST_F(TEST_TOPK_PQ_DISTANCE_UT, DATA_TYPE_FLOAT_TIME_5INPUT) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_INT32};
  std::string order = "DES";
  constexpr int32_t k = 1024;
  constexpr int32_t group_size = 60;  
  constexpr int32_t actual_count0 = 240000;

  vector<float> expect_vec;

  vector<vector<int64_t>> shapes = {{actual_count0}, {k}, {actual_count0/group_size}, {}, {actual_count0}, {actual_count0/group_size}};
  float pq_distance0[actual_count0];
  float grouped_extreme_distance0[actual_count0/group_size];
  int32_t pq_ivf0[actual_count0];
  int32_t pq_index0[actual_count0];

  for (int32_t i = 0; i < actual_count0; i++) {
    pq_distance0[i] = (rand() % 240000 + 1);
    pq_ivf0[i] = i;
    pq_index0[i] = 1;
    expect_vec.emplace_back(pq_distance0[i]);
  }
  fillExtreme(grouped_extreme_distance0, pq_distance0, actual_count0, group_size, order);

  // ouput
  float topk_distance[k];
  int32_t topk_ivf[k];
  int32_t topk_index[k];
  vector<void*> datas = {(void*)(&actual_count0),
                         (void*)pq_distance0,
                         (void*)grouped_extreme_distance0,
                         (void*)pq_ivf0,
                         (void*)pq_index0,
                         (void*)topk_distance,
                         (void*)topk_ivf,
                         (void*)topk_index};
  CREATE_NODEDEF2(shapes, data_types, datas, k, group_size);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  sort(expect_vec.begin(), expect_vec.end(), [order](float a, float b) {
    if (order == "DES") {
      return a > b;
    } else { 
      return a < b;
    } 
  });
 
  for (int i = 0; i < k; i++) {
    EXPECT_EQ(expect_vec[i], topk_distance[i]);
  } 
} 


TEST_F(TEST_TOPK_PQ_DISTANCE_UT, DATA_TYPE_FLOAT16_TIME_6INPUT) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_INT32};
  std::string order = "DES";
  constexpr int32_t k = 1024;
  constexpr int32_t group_size = 60;  
  constexpr int32_t actual_count0 = 240000;

  vector<Eigen::half> expect_vec;

  vector<vector<int64_t>> shapes = {{actual_count0}, {k}, {actual_count0/group_size}, {}, {actual_count0}, {actual_count0/group_size}};
  Eigen::half pq_distance0[actual_count0];
  Eigen::half grouped_extreme_distance0[actual_count0/group_size];
  int32_t pq_ivf0[actual_count0];
  int32_t pq_index0[actual_count0];

  for (int32_t i = 0; i < actual_count0; i++) {
    pq_distance0[i] = Eigen::half(rand() % 24000 + 1);
    pq_ivf0[i] = i;
    pq_index0[i] = 1;
    expect_vec.emplace_back(pq_distance0[i]);
  }
  fillExtreme(grouped_extreme_distance0, pq_distance0, actual_count0, group_size, order);

  // ouput
  Eigen::half topk_distance[k];
  int32_t topk_ivf[k];
  int32_t topk_index[k];
  vector<void*> datas = {(void*)(&actual_count0),
                         (void*)pq_distance0,
                         (void*)grouped_extreme_distance0,
                         (void*)pq_ivf0,
                         (void*)pq_index0,
                         (void*)topk_distance,
                         (void*)topk_ivf,
                         (void*)topk_index};
  CREATE_NODEDEF2(shapes, data_types, datas, k, group_size);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  sort(expect_vec.begin(), expect_vec.end(), [order](Eigen::half a, Eigen::half b) {
    if (order == "DES") {
      return a > b;
    } else { 
      return a < b;
    } 
  });
 
  for (int i = 0; i < k; i++) {
    EXPECT_EQ(expect_vec[i], topk_distance[i]);
  } 
} 