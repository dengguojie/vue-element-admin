#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "unsupported/Eigen/CXX11/Tensor"
#undef private
#undef protected
#include <cmath>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_MULTINOMIAL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, seed1, seed2)           \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();        \
  if ((seed1) == -1) {                                                    \
    NodeDefBuilder(node_def.get(), "Multinomial", "Multinomial")          \
        .Input({"logits", (data_types)[0], (shapes)[0], (datas)[0]})      \
        .Input({"num_samples", (data_types)[1], (shapes)[1], (datas)[1]}) \
        .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})          \
        .Attr("dtype", (data_types)[3]);                                  \
  } else if ((seed2) == -1) {                                             \
    NodeDefBuilder(node_def.get(), "Multinomial", "Multinomial")          \
        .Input({"logits", (data_types)[0], (shapes)[0], (datas)[0]})      \
        .Input({"num_samples", (data_types)[1], (shapes)[1], (datas)[1]}) \
        .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})          \
        .Attr("seed", (seed1))                                            \
        .Attr("dtype", (data_types)[3]);                                  \
  } else {                                                                \
    NodeDefBuilder(node_def.get(), "Multinomial", "Multinomial")          \
        .Input({"logits", (data_types)[0], (shapes)[0], (datas)[0]})      \
        .Input({"num_samples", (data_types)[1], (shapes)[1], (datas)[1]}) \
        .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})          \
        .Attr("seed", (seed1))                                            \
        .Attr("seed2", (seed2))                                           \
        .Attr("dtype", (data_types)[3]);                                  \
  }

#define MULTINOMIAL_CASE(case_name, dtypes, shapes, datas, seed1, seed2, status) \
  TEST_F(TEST_MULTINOMIAL_UT, TestMultinomial_##case_name) {                     \
    CREATE_NODEDEF(shapes, dtypes, datas, seed1, seed2);                         \
    RUN_KERNEL(node_def, HOST, status);                                          \
  }

namespace {
const int num_examples = 100;
int32_t input_1_num = num_examples;

vector<DataType> dtypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32};
vector<vector<int64_t>> shapes = {{2, 3}, {}, {2, num_examples}};
float input_0[6] = {1, 1, 1, 1, 3, 3};
int32_t* input_1 = &input_1_num;
int32_t output[2 * num_examples] = {0};
}  // namespace

// success seed & float
namespace {
vector<void*> datas = {(void*)input_0, (void*)input_1, (void*)output};
vector<DataType> dtypes_int64_out = {DT_FLOAT, DT_INT32, DT_INT64, DT_INT64};
int64_t output_int64[2 * num_examples] = {0};
vector<void*> datas_int64_out = {(void*)input_0, (void*)input_1,
                                 (void*)output_int64};
const int32_t num_examples_multi = 20 * 1025;
int32_t input_1_num_multi = num_examples_multi;
int32_t* input_1_multi = &input_1_num_multi;
vector<vector<int64_t>> shapes_multi = {{2, 3}, {}, {2, 20500}};
int32_t output_multi[40 * 1025] = {0};
vector<void*> datas_multi = {(void*)input_0, (void*)input_1_multi,
                             (void*)output_multi};

double input_0_double[] = {1, 1, 1, 1, 3, 3};
vector<DataType> dtypes_int64_out2 = {DT_DOUBLE, DT_INT32, DT_INT64, DT_INT64};
int64_t output_multi_int64[40 * 1025] = {0};
vector<void*> datas_multi_double_int64 = {
    (void*)input_0_double, (void*)input_1_multi, (void*)output_multi_int64};
vector<void*> datas_multi_double = {(void*)input_0_double, (void*)input_1_multi,
                                    (void*)output_multi};

}  // namespace

MULTINOMIAL_CASE(succ_float_int32_int32_rand_seed, dtypes, shapes, datas, 0, 0,
                 KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_float_int32_int32_seed1, dtypes, shapes, datas, 2, 0,
                 KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_float_int32_int32_seed1_2, dtypes, shapes, datas, 2, 3,
                 KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_float_int32_int32_seed2, dtypes, shapes, datas, 0, 3,
                 KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_float_int32_int64, dtypes_int64_out, shapes,
                 datas_int64_out, 0, 3, KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_float_int32_int32_multi, dtypes, shapes_multi,
                 datas_multi, 0, 0, KERNEL_STATUS_OK)

// successs half & double
namespace {
vector<DataType> dtypes_half = {DT_FLOAT16, DT_INT32, DT_INT32, DT_INT32};
Eigen::half input_0_half[] = {Eigen::half(1), Eigen::half(1),  Eigen::half(1),
                              Eigen::half(1), Eigen::half(10), Eigen::half(10)};
vector<void*> datas_half = {(void*)input_0_half, (void*)input_1, (void*)output};

vector<DataType> dtypes_half_int64 = {DT_FLOAT16, DT_INT32, DT_INT64, DT_INT64};
vector<void*> datas_half_multi_int64 = {
    (void*)input_0_half, (void*)input_1_multi, (void*)output_multi_int64};
vector<void*> datas_half_multi = {(void*)input_0_half, (void*)input_1_multi,
                                  (void*)output_multi};

}  // namespace
MULTINOMIAL_CASE(succ_half_int32_int32, dtypes_half, shapes, datas_half, 0, 0,
                 KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_half_int32_int32_multi, dtypes_half, shapes_multi,
                 datas_half_multi, 0, 0, KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_half_int32_int64_multi, dtypes_half_int64, shapes_multi,
                 datas_half_multi_int64, 0, 0, KERNEL_STATUS_OK)

namespace {
vector<DataType> dtypes_double = {DT_DOUBLE, DT_INT32, DT_INT32, DT_INT32};
vector<void*> datas_double = {(void*)input_0_double, (void*)input_1,
                              (void*)output};

int64_t output_int64_2[2 * num_examples] = {0};
vector<void*> datas_int64_out_2 = {(void*)input_0, (void*)input_1,
                                   (void*)output_int64_2};
}  // namespace
MULTINOMIAL_CASE(succ_double_int32_int32, dtypes_double, shapes, datas_double,
                 0, 0, KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_double_int32_int32_multi, dtypes_double, shapes_multi,
                 datas_multi_double, 0, 0, KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_double_int32_int64, dtypes_int64_out2, shapes,
                 datas_int64_out_2, 0, 0, KERNEL_STATUS_OK)
MULTINOMIAL_CASE(succ_double_int32_int64_multi, dtypes_int64_out2, shapes_multi,
                 datas_multi_double_int64, 0, 0, KERNEL_STATUS_OK)

// fail dtype
namespace {
vector<DataType> dtypes_e1 = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
vector<DataType> dtypes_e2 = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_INT32};
}  // namespace
MULTINOMIAL_CASE(fail_input0_int32, dtypes_e1, shapes, datas, 0, 0,
                 KERNEL_STATUS_PARAM_INVALID)
MULTINOMIAL_CASE(fail_input1_float, dtypes_e2, shapes, datas, 0, 0,
                 KERNEL_STATUS_PARAM_INVALID)

// fail shape
namespace {
vector<vector<int64_t>> shapes_e1 = {{6}, {}, {2, num_examples}};
vector<vector<int64_t>> shapes_e3 = {{6}, {1}, {2, num_examples}};
float wrong_input_1[1] = {100};
vector<void*> datas_e1 = {(void*)input_0, (void*)wrong_input_1, (void*)output};
}  // namespace
MULTINOMIAL_CASE(fail_input0_shape, dtypes, shapes_e1, datas, 0, 0,
                 KERNEL_STATUS_PARAM_INVALID)
MULTINOMIAL_CASE(fail_input1_shape, dtypes, shapes_e3, datas_e1, 0, 0,
                 KERNEL_STATUS_PARAM_INVALID)

// fail nonpositive class num
namespace {
vector<vector<int64_t>> shapes_e2 = {{0, 0}, {}, {2, num_examples}};
}
MULTINOMIAL_CASE(fail_input0_zero_class_num, dtypes, shapes_e2, datas, 0, 0,
                 KERNEL_STATUS_PARAM_INVALID)

// fail negetive example num
namespace {
int32_t input_1_num_e1 = -1;
int32_t* input_1_e1 = &input_1_num_e1;
vector<void*> datas_e2 = {(void*)input_0, (void*)input_1_e1, (void*)output};
}  // namespace
MULTINOMIAL_CASE(fail_input1_negative_example_num, dtypes, shapes, datas_e2, 0,
                 0, KERNEL_STATUS_PARAM_INVALID)

// success zero example num
namespace {
int32_t input_1_num_e2 = 0;
int32_t* input_1_e2 = &input_1_num_e2;
vector<void*> datas_e3 = {(void*)input_0, (void*)input_1_e2, (void*)output};
}  // namespace
MULTINOMIAL_CASE(succ_input1_zero_examples_to_generate, dtypes, shapes,
                 datas_e3, 0, 0, KERNEL_STATUS_OK)