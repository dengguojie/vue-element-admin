#include <iostream>
#include <sstream>

#include "Eigen/Core"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "gtest/gtest.h"
#include "node_def_builder.h"

using namespace std;
using namespace aicpu;

class TEST_STATELESS_DROP_OUT_GEN_MASK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "StatelessDropOutGenMask",        \
                 "StatelessDropOutGenMask")                        \
      .Input({"shape", data_types[0], shapes[0], datas[0]})        \
      .Input({"prob", data_types[1], shapes[1], datas[1]})         \
      .Input({"seed", data_types[2], shapes[2], datas[2]})         \
      .Input({"seed1", data_types[3], shapes[3], datas[3]})        \
      .Output({"y", data_types[4], shapes[4], datas[4]})

#define ADD_CASE(shape_base_type, shape_aicpu_type, prob_base_type,      \
                 prob_aicpu_type, seed_base_type, seed_aicpu_type,       \
                 seed1_base_type, seed1_aicpu_type)                      \
  TEST_F(TEST_STATELESS_DROP_OUT_GEN_MASK_UT,                            \
         TestStatelessDropOutGenMask_##prob_aicpu_type) {                \
    vector<DataType> data_types = {shape_aicpu_type, prob_aicpu_type,    \
                                   seed_aicpu_type, seed1_aicpu_type,    \
                                   DT_UINT8};                            \
    vector<vector<int64_t>> shapes = {{4, 4}, {0}, {0}, {0}, {16}};      \
    shape_base_type shape[] = {4, 4};                                    \
    prob_base_type prob = (prob_base_type)0.7;                           \
    seed_base_type seed = (seed_base_type)126;                           \
    seed1_base_type seed1 = (seed1_base_type)0;                          \
    uint8_t out[16] = {0};                                               \
    vector<void *> datas = {(void *)shape, (void *)&prob, (void *)&seed, \
                            (void *)&seed1, (void *)out};                \
    CREATE_NODEDEF(shapes, data_types, datas);                           \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    for (int32_t i = 0; i < 16; i++) {                                   \
      printf("%u ", out[i]);                                             \
    }                                                                    \
    printf("\n");                                                        \
  }

#define ADD_PROB_ZERO_CASE(shape_base_type, shape_aicpu_type, prob_base_type, \
                      prob_aicpu_type, seed_base_type, seed_aicpu_type,  \
                      seed1_base_type, seed1_aicpu_type)                 \
  TEST_F(TEST_STATELESS_DROP_OUT_GEN_MASK_UT,                            \
         TestStatelessDropOutGenMask_prob_zero_##prob_aicpu_type) {           \
    vector<DataType> data_types = {shape_aicpu_type, prob_aicpu_type,    \
                                   seed_aicpu_type, seed1_aicpu_type,    \
                                   DT_UINT8};                            \
    vector<vector<int64_t>> shapes = {{4, 4}, {0}, {0}, {0}, {16}};      \
    shape_base_type shape[] = {4, 4};                                    \
    prob_base_type prob = (prob_base_type)0;                             \
    seed_base_type seed = (seed_base_type)126;                           \
    seed1_base_type seed1 = (seed1_base_type)0;                          \
    uint8_t out[16] = {0};                                               \
    vector<void *> datas = {(void *)shape, (void *)&prob, (void *)&seed, \
                            (void *)&seed1, (void *)out};                \
    CREATE_NODEDEF(shapes, data_types, datas);                           \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    for (int32_t i = 0; i < 16; i++) {                                   \
      printf("%u ", out[i]);                                             \
    }                                                                    \
    printf("\n");                                                        \
  }

#define ADD_PROB_ONE_CASE(shape_base_type, shape_aicpu_type, prob_base_type, \
                      prob_aicpu_type, seed_base_type, seed_aicpu_type,  \
                      seed1_base_type, seed1_aicpu_type)                 \
  TEST_F(TEST_STATELESS_DROP_OUT_GEN_MASK_UT,                            \
         TestStatelessDropOutGenMask_prob_one_##prob_aicpu_type) {           \
    vector<DataType> data_types = {shape_aicpu_type, prob_aicpu_type,    \
                                   seed_aicpu_type, seed1_aicpu_type,    \
                                   DT_UINT8};                            \
    vector<vector<int64_t>> shapes = {{4, 4}, {0}, {0}, {0}, {16}};      \
    shape_base_type shape[] = {4, 4};                                    \
    prob_base_type prob = (prob_base_type)1;                             \
    seed_base_type seed = (seed_base_type)126;                           \
    seed1_base_type seed1 = (seed1_base_type)0;                          \
    uint8_t out[16] = {0};                                               \
    vector<void *> datas = {(void *)shape, (void *)&prob, (void *)&seed, \
                            (void *)&seed1, (void *)out};                \
    CREATE_NODEDEF(shapes, data_types, datas);                           \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    for (int32_t i = 0; i < 16; i++) {                                   \
      printf("%u ", out[i]);                                             \
    }                                                                    \
    printf("\n");                                                        \
  }

ADD_CASE(int32_t, DT_INT32, Eigen::half, DT_FLOAT16, int32_t, DT_INT32, int32_t,
         DT_INT32)
ADD_CASE(int64_t, DT_INT64, float, DT_FLOAT, int64_t, DT_INT64, int64_t,
         DT_INT64)
ADD_PROB_ZERO_CASE(int32_t, DT_INT32, float, DT_FLOAT, int64_t, DT_INT64, int64_t,
         DT_INT64)
ADD_PROB_ONE_CASE(int32_t, DT_INT32, float, DT_FLOAT, int64_t, DT_INT64, int64_t,
         DT_INT64)