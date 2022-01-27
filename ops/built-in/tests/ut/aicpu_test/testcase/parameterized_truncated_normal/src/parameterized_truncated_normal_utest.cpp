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
using Eigen::half;

class TEST_PARAMETERIZED_TRUNCATED_NORMAL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, seed1, seed2)                                    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                                 \
  if ((seed1) == -1) {                                                                             \
    NodeDefBuilder(node_def.get(), "ParameterizedTruncatedNormal", "ParameterizedTruncatedNormal") \
        .Input({"shape", (data_types)[0], (shapes)[0], (datas)[0]})                                \
        .Input({"means", (data_types)[1], (shapes)[1], (datas)[1]})                                \
        .Input({"stdevs", (data_types)[2], (shapes)[2], (datas)[2]})                               \
        .Input({"min", (data_types)[3], (shapes)[3], (datas)[3]})                                  \
        .Input({"max", (data_types)[4], (shapes)[4], (datas)[4]})                                  \
        .Output({"y", (data_types)[5], (shapes)[5], (datas)[5]});                                  \
  } else if ((seed2) == -1) {                                                                      \
    NodeDefBuilder(node_def.get(), "ParameterizedTruncatedNormal", "ParameterizedTruncatedNormal") \
        .Input({"shape", (data_types)[0], (shapes)[0], (datas)[0]})                                \
        .Input({"means", (data_types)[1], (shapes)[1], (datas)[1]})                                \
        .Input({"stdevs", (data_types)[2], (shapes)[2], (datas)[2]})                               \
        .Input({"min", (data_types)[3], (shapes)[3], (datas)[3]})                                  \
        .Input({"max", (data_types)[4], (shapes)[4], (datas)[4]})                                  \
        .Output({"y", (data_types)[5], (shapes)[5], (datas)[5]})                                   \
        .Attr("seed", seed1);                                                                      \
  } else {                                                                                         \
    NodeDefBuilder(node_def.get(), "ParameterizedTruncatedNormal", "ParameterizedTruncatedNormal") \
        .Input({"shape", (data_types)[0], (shapes)[0], (datas)[0]})                                \
        .Input({"means", (data_types)[1], (shapes)[1], (datas)[1]})                                \
        .Input({"stdevs", (data_types)[2], (shapes)[2], (datas)[2]})                               \
        .Input({"min", (data_types)[3], (shapes)[3], (datas)[3]})                                  \
        .Input({"max", (data_types)[4], (shapes)[4], (datas)[4]})                                  \
        .Output({"y", (data_types)[5], (shapes)[5], (datas)[5]})                                   \
        .Attr("seed", seed1)                                                                       \
        .Attr("seed2", seed2);                                                                     \
  }

#define PATAMETERIZED_TRUNCATED_NORMAL_CASE(case_name, dtypes, shapes, datas, seed1, seed2, status) \
  TEST_F(TEST_PARAMETERIZED_TRUNCATED_NORMAL_UT, case_name) {                                       \
    CREATE_NODEDEF(shapes, dtypes, datas, seed1, seed2);                                            \
    RUN_KERNEL(node_def, HOST, status);                                                             \
  }

namespace {
const int num_per_batch = 100;
const int batch_size = 3;

int32_t sample_shape[2] = {batch_size, num_per_batch};
float means[3] = {0, 10, 100};
float stdevs[3] = {1, 1, 1};
float mins[3] = {-0.5, 10 - 0.5, 100 - 2};
float maxs[3] = {1.5, 10 + 1.2, 100 - 1};
float y[batch_size * num_per_batch] = {0};
vector<DataType> dtypes = {DT_INT32, DT_FLOAT, DT_FLOAT,
                           DT_FLOAT, DT_FLOAT, DT_FLOAT};
vector<vector<int64_t>> shapes = {{2}, {3}, {3},
                                  {3}, {3}, {batch_size, num_per_batch}};
}  // namespace

// success seed & float
namespace {
vector<void *> datas = {(void *)sample_shape, (void *)means, (void *)stdevs,
                        (void *)mins,         (void *)maxs,  (void *)y};
vector<DataType> dtypes_int64 = {DT_INT64, DT_FLOAT, DT_FLOAT,
                                 DT_FLOAT, DT_FLOAT, DT_FLOAT};
int64_t sample_shape_int64[2] = {batch_size, num_per_batch};
vector<void *> datas_int64 = {(void *)sample_shape_int64,
                              (void *)means,
                              (void *)stdevs,
                              (void *)mins,
                              (void *)maxs,
                              (void *)y};
}  // namespace

PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_float_rand_seed, dtypes, shapes,
                                    datas, 0, 0, KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_float_seed1, dtypes, shapes,
                                    datas, 2, 0, KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_float_seed1_2, dtypes, shapes,
                                    datas, 2, 3, KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_float_seed2, dtypes, shapes,
                                    datas, 0, 3, KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int64_float, dtypes_int64, shapes,
                                    datas_int64, 0, 3, KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int64_float2, dtypes_int64, shapes,
                                    datas_int64, 0, 3, KERNEL_STATUS_OK)

// successs half
namespace {
vector<DataType> dtypes_half = {DT_INT32,   DT_FLOAT16, DT_FLOAT16,
                                DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};

half means_h[3] = {half(0), half(10), half(100)};
half stdevs_h[3] = {half(1), half(1), half(1)};
half mins_h[3] = {half(-0.5), half(10 - 0.5), half(100 - 2)};
half maxs_h[3] = {half(1.5), half(10 + 1.2), half(100 - 1)};
half y_h[batch_size * num_per_batch];

vector<void *> datas_half = {(void *)sample_shape, (void *)means_h,
                             (void *)stdevs_h,     (void *)mins_h,
                             (void *)maxs_h,       (void *)y_h};

vector<void *> datas_half_int64 = {(void *)sample_shape_int64,
                                   (void *)means_h,
                                   (void *)stdevs_h,
                                   (void *)mins_h,
                                   (void *)maxs_h,
                                   (void *)y_h};
vector<DataType> dtypes_half_int64 = {DT_INT64,   DT_FLOAT16, DT_FLOAT16,
                                      DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
}  // namespace
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_half, dtypes_half, shapes,
                                    datas_half, 0, 0, KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int64_half, dtypes_half_int64, shapes,
                                    datas_half_int64, 0, 0, KERNEL_STATUS_OK)

// successs double
namespace {
double means_d[3] = {0, 10, 100};
double stdevs_d[3] = {1, 1, 1};
double mins_d[3] = {-0.5, 10 - 0.5, 100 - 2};
double maxs_d[3] = {1.5, 10 + 1.2, 100 - 1};
double y_d[batch_size * num_per_batch] = {0};
vector<DataType> dtypes_double = {DT_INT32,  DT_DOUBLE, DT_DOUBLE,
                                  DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
vector<void *> datas_double = {(void *)sample_shape, (void *)means_d,
                               (void *)stdevs_d,     (void *)mins_d,
                               (void *)maxs_d,       (void *)y_d};
vector<void *> datas_double_int64 = {(void *)sample_shape_int64,
                                     (void *)means_d,
                                     (void *)stdevs_d,
                                     (void *)mins_d,
                                     (void *)maxs_d,
                                     (void *)y_d};
vector<DataType> dtypes_double_int64 = {DT_INT64,  DT_DOUBLE, DT_DOUBLE,
                                        DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
}  // namespace
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_double, dtypes_double, shapes,
                                    datas_double, 0, 0, KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int64_double, dtypes_double_int64,
                                    shapes, datas_double_int64, 0, 0,
                                    KERNEL_STATUS_OK)

// falure datatype
namespace {
vector<DataType> dtypes_float_double = {DT_INT32, DT_FLOAT, DT_DOUBLE,
                                        DT_FLOAT, DT_FLOAT, DT_FLOAT};
vector<void *> datas_float_double = {(void *)sample_shape, (void *)means,
                                     (void *)stdevs_d,     (void *)mins,
                                     (void *)maxs,         (void *)y};

vector<DataType> dtypes_allfloat = {DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                    DT_FLOAT, DT_FLOAT, DT_FLOAT};
float sample_shape_float[2] = {batch_size, num_per_batch};
vector<void *> datas_allfloat_double = {(void *)sample_shape_float,
                                        (void *)means,
                                        (void *)stdevs,
                                        (void *)mins,
                                        (void *)maxs,
                                        (void *)y};
}  // namespace
PATAMETERIZED_TRUNCATED_NORMAL_CASE(fail_int32_float_double,
                                    dtypes_float_double, shapes,
                                    datas_float_double, 0, 0,
                                    KERNEL_STATUS_PARAM_INVALID)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(fail_allfloat, dtypes_allfloat, shapes,
                                    datas_allfloat_double, 0, 0,
                                    KERNEL_STATUS_PARAM_INVALID)

// falure shapes
namespace {
int32_t sample_shape_negative_batch[2] = {-batch_size, num_per_batch};
int32_t sample_shape_negative_dimension[2] = {batch_size, -num_per_batch};

vector<void *> datas_negative_batch = {(void *)sample_shape_negative_batch,
                                       (void *)means,
                                       (void *)stdevs,
                                       (void *)mins,
                                       (void *)maxs,
                                       (void *)y};
vector<void *> datas_negative_dimension = {
    (void *)sample_shape_negative_dimension,
    (void *)means,
    (void *)stdevs,
    (void *)mins,
    (void *)maxs,
    (void *)y};
}  // namespace
PATAMETERIZED_TRUNCATED_NORMAL_CASE(fail_negative_batch, dtypes, shapes,
                                    datas_negative_batch, 0, 0,
                                    KERNEL_STATUS_PARAM_INVALID)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(fail_negative_dimension, dtypes, shapes,
                                    datas_negative_dimension, 0, 0,
                                    KERNEL_STATUS_PARAM_INVALID)

// pseudo-broadcast
namespace {
float zero = 0;
float negative_half = -0.5;
float *means_scalar = &zero;
float *mins_scalar = &negative_half;
float stdevs_bcast[1] = {1};
float mins_bcast[1] = {-0.5f};
float maxs_bcast[1] = {200};

float means_bcast_err[2] = {0, 10};
vector<void *> datas_scalar = {(void *)sample_shape, (void *)means_scalar,
                               (void *)stdevs,       (void *)mins_scalar,
                               (void *)maxs,         (void *)y};
vector<void *> datas_bcast = {(void *)sample_shape, (void *)means,
                              (void *)stdevs_bcast, (void *)mins_bcast,
                              (void *)maxs_bcast,   (void *)y};
vector<void *> datas_bcast_err = {(void *)sample_shape, (void *)means_bcast_err,
                                  (void *)stdevs,       (void *)mins,
                                  (void *)maxs,         (void *)y};
vector<vector<int64_t>> shapes_bcast_scalar = {
    {2}, {}, {3}, {3}, {3}, {batch_size, num_per_batch}};
vector<vector<int64_t>> shapes_bcast = {{2}, {1}, {3},
                                        {3}, {3}, {batch_size, num_per_batch}};
vector<vector<int64_t>> shapes_bcast_err = {
    {2}, {2}, {3}, {3}, {3}, {batch_size, num_per_batch}};

}  // namespace
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_float_scalar, dtypes,
                                    shapes_bcast_scalar, datas_scalar, 0, 0,
                                    KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(succ_int32_float_bcast, dtypes,
                                    shapes_bcast, datas_bcast, 0, 0,
                                    KERNEL_STATUS_OK)
PATAMETERIZED_TRUNCATED_NORMAL_CASE(fail_int32_float_bcast, dtypes,
                                    shapes_bcast_err, datas_bcast_err, 0, 0,
                                    KERNEL_STATUS_PARAM_INVALID)