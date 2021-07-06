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

class TEST_INPLACE_TOPK_DISTANCE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "InplaceTopKDistance", "InplaceTopKDistance") \
      .Input({"topk_pq_distance", data_types[0], shapes[0], datas[0]})         \
      .Input({"topk_pq_index", data_types[1], shapes[0], datas[1]})            \
      .Input({"topk_pq_ivf", data_types[2], shapes[0], datas[2]})              \
      .Input({"pq_distance", data_types[0], shapes[1], datas[3]})              \
      .Input({"pq_index", data_types[1], shapes[1], datas[4]})                 \
      .Input({"pq_ivf", data_types[3], shapes[2], datas[5]})                   \
      .Attr("order", "asc")

template <typename T>
void printData(T* data, int32_t size) {
  for (int32_t i = 0; i < size; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

TEST_F(TEST_INPLACE_TOPK_DISTANCE_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10}, {6}, {}};
  float topk_pq_distance[10];
  int32_t topk_pq_index[10];
  int32_t topk_pq_ivf[10];
  for (int32_t i = 0; i < 10; i++) {
    topk_pq_distance[i] = i * 1.0;
    topk_pq_index[i] = i;
    topk_pq_ivf[i] = 3;
  }
  float pq_distance[6];
  int32_t pq_index[6];
  int32_t pq_ivf = 9;
  for (int32_t i = 0; i < 6; i++) {
    pq_distance[i] = 8 + i;
    pq_index[i] = 7 + i;
  }

  std::cout << "==========before comput=========" << std::endl;
  printData<float>(topk_pq_distance, 10);
  printData<int32_t>(topk_pq_index, 10);
  printData<int32_t>(topk_pq_ivf, 10);

  printData<float>(pq_distance, 6);
  printData<int32_t>(pq_index, 6);
  std::cout << pq_ivf << std::endl;

  vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                         (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::cout << "==========after comput=========" << std::endl;
  printData<float>(topk_pq_distance, 10);
  printData<int32_t>(topk_pq_index, 10);
  printData<int32_t>(topk_pq_ivf, 10);
}

TEST_F(TEST_INPLACE_TOPK_DISTANCE_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10}, {6}, {}};
  Eigen::half topk_pq_distance[10];
  int32_t topk_pq_index[10];
  int32_t topk_pq_ivf[10];
  for (int32_t i = 0; i < 10; i++) {
    topk_pq_distance[i] = Eigen::half(i * 1.1);
    topk_pq_index[i] = i;
    topk_pq_ivf[i] = 3;
  }
  Eigen::half pq_distance[6];
  int32_t pq_index[6];
  int32_t pq_ivf = 9;
  for (int32_t i = 0; i < 6; i++) {
    pq_distance[i] = Eigen::half(7.3 + i);
    pq_index[i] = 7 + i;
  }

  std::cout << "==========DT_FLOAT16 before comput=========" << std::endl;
  printData<Eigen::half>(topk_pq_distance, 10);
  printData<int32_t>(topk_pq_index, 10);
  printData<int32_t>(topk_pq_ivf, 10);

  printData<Eigen::half>(pq_distance, 6);
  printData<int32_t>(pq_index, 6);
  std::cout << pq_ivf << std::endl;

  vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                         (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::cout << "==========DT_FLOAT16 after comput=========" << std::endl;
  printData<Eigen::half>(topk_pq_distance, 10);
  printData<int32_t>(topk_pq_index, 10);
  printData<int32_t>(topk_pq_ivf, 10);
}

TEST_F(TEST_INPLACE_TOPK_DISTANCE_UT, DATA_TYPE_FLOAT_LARG_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_INT32, DT_INT32};
  const int32_t topk_size = 10;
  const int32_t pq_size = 20;

  vector<vector<int64_t>> shapes = {{topk_size}, {pq_size}, {}};
  Eigen::half topk_pq_distance[topk_size];
  int32_t topk_pq_index[topk_size];
  int32_t topk_pq_ivf[topk_size];
  for (int32_t i = 0; i < topk_size; i++) {
    topk_pq_distance[i] = Eigen::half(i * 1.1);
    topk_pq_index[i] = i;
    topk_pq_ivf[i] = 3;
  }

  Eigen::half pq_distance[pq_size];
  int32_t pq_index[pq_size];
  int32_t pq_ivf = 8;
  for (int32_t i = 0; i < pq_size; i++) {
    pq_distance[i] = Eigen::half(5.1 + i);
    pq_index[i] = 7 + i;
  }

  std::cout << "==========DATA_TYPE_FLOAT_LARG_SUCC before comput=========" << std::endl;
  printData<Eigen::half>(topk_pq_distance, topk_size);
  printData<int32_t>(topk_pq_index, topk_size);
  printData<int32_t>(topk_pq_ivf, topk_size);

  printData<Eigen::half>(pq_distance, pq_size);
  printData<int32_t>(pq_index, pq_size);
  std::cout << pq_ivf << std::endl;

  vector<void*> datas = {(void*)topk_pq_distance, (void*)topk_pq_index, (void*)topk_pq_ivf,
                         (void*)pq_distance,      (void*)pq_index,      (void*)&pq_ivf};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::cout << "==========DATA_TYPE_FLOAT_LARG_SUCC after comput=========" << std::endl;
  printData<Eigen::half>(topk_pq_distance, topk_size);
  printData<int32_t>(topk_pq_index, topk_size);
  printData<int32_t>(topk_pq_ivf, topk_size);
}