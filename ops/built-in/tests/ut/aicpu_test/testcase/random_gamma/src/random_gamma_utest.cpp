#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_test_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_RANDOM_GAMMA_UT : public testing::Test
{
};

#define CREATE_NODEDEF(shapes, data_types, datas, seed1, seed2)    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  if (seed1 == -1)                                                 \
  {                                                                \
    NodeDefBuilder(node_def.get(), "RandomGamma", "RandomGamma")   \
        .Input({"shape", data_types[0], shapes[0], datas[0]})      \
        .Input({"alpha", data_types[1], shapes[1], datas[1]})      \
        .Output({"y", data_types[1], shapes[2], datas[2]});        \
  }                                                                \
  else if (seed2 == -1)                                            \
  {                                                                \
    NodeDefBuilder(node_def.get(), "RandomGamma", "RandomGamma")   \
        .Input({"shape", data_types[0], shapes[0], datas[0]})      \
        .Input({"alpha", data_types[1], shapes[1], datas[1]})      \
        .Output({"y", data_types[1], shapes[2], datas[2]})         \
        .Attr("seed", seed1);                                      \
  }                                                                \
  else                                                             \
  {                                                                \
    NodeDefBuilder(node_def.get(), "RandomGamma", "RandomGamma")   \
        .Input({"shape", data_types[0], shapes[0], datas[0]})      \
        .Input({"alpha", data_types[1], shapes[1], datas[1]})      \
        .Output({"y", data_types[1], shapes[2], datas[2]})         \
        .Attr("seed", seed1)                                       \
        .Attr("seed2", seed2);                                     \
  }

#define RANDOM_GAMMA_CASE_WITH_TYPE(case_name, base_type, aicpu_data_type,              \
                                    seed1, seed2)                                       \
  TEST_F(TEST_RANDOM_GAMMA_UT, TestRandomGamma_##case_name)                             \
  {                                                                                     \
    vector<DataType> data_types = {DT_INT64, aicpu_data_type};                          \
    int64_t input1[1] = {(int64_t)10};                                                  \
    base_type input2[4] = {(base_type)2, (base_type)3, (base_type)0.4, (base_type)0.8}; \
    base_type output[40] = {(base_type)0};                                              \
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};            \
    vector<vector<int64_t>> shapes = {{input1[0]}, {4}, {input1[0], 4}};                \
    CREATE_NODEDEF(shapes, data_types, datas, seed1, seed2);                            \
    if (aicpu_data_type != DT_FLOAT && aicpu_data_type != DT_DOUBLE &&                  \
        aicpu_data_type != DT_FLOAT16)                                                  \
    {                                                                                   \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                          \
    }                                                                                   \
    else                                                                                \
    {                                                                                   \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                     \
    }                                                                                   \
  }

RANDOM_GAMMA_CASE_WITH_TYPE(float16_with_seed, Eigen::half, DT_FLOAT16, 10, -1)

RANDOM_GAMMA_CASE_WITH_TYPE(float_with_seed2, float, DT_FLOAT, 10, 5)

RANDOM_GAMMA_CASE_WITH_TYPE(double_with_no_seed, double, DT_DOUBLE, -1, -1)

RANDOM_GAMMA_CASE_WITH_TYPE(data_type_not_support, int64_t, DT_INT64, -1, -1)
