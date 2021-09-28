#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/allocator_utils.h"
#undef private
#undef protected
#include <algorithm>

#include "Eigen/Core"

using namespace std;
using namespace aicpu;

namespace {
uint64_t result_summary[4] = {0};
}

class TEST_NON_MAX_SUPPRESSION_V3_UT : public testing::Test {};

#define CREATE_NODEDEF_MEMCOPY(shapes, data_types, datas)                 \
  auto memcpy_node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(memcpy_node_def.get(), "MemCopy", "MemCopy")             \
      .Input({"release_flag", data_types[0], shapes[0], datas[0]})        \
      .Input({"data_size", data_types[1], shapes[1], datas[1]})           \
      .Input({"src_ptr", data_types[2], shapes[2], datas[2]})             \
      .Input({"dst_ptr", data_types[3], shapes[3], datas[3]})             \
      .Attr("num", 2);

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3") \
      .Input({"boxes", data_types[0], shapes[0], datas[0]})                    \
      .Input({"scores", data_types[1], shapes[1], datas[1]})                   \
      .Input({"max_output_size", data_types[2], shapes[2], datas[2]})          \
      .Input({"iou_threshold", data_types[3], shapes[3], datas[3]})            \
      .Input({"score_threshold", data_types[4], shapes[4], datas[4]})          \
      .Output({"result_summary", data_types[5], shapes[5], datas[5]})

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT,                                       \
         TestNonMaxSuppressionV3_##aicpu_type) {                               \
    vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32,           \
                                   aicpu_type, aicpu_type, DT_UINT64};         \
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};       \
    base_type boxes[] = {                                                      \
        (base_type)24.664701, (base_type)92.11955,  (base_type)6.000731,       \
        (base_type)50.0787,   (base_type)83.716705, (base_type)3.6205719,      \
        (base_type)9.723194,  (base_type)46.209377, (base_type)77.11508,       \
        (base_type)74.02599,  (base_type)78.719894, (base_type)31.890612,      \
        (base_type)16.0346,   (base_type)21.073357, (base_type)31.530176,      \
        (base_type)54.47457,  (base_type)45.646084, (base_type)26.065968,      \
        (base_type)30.528221, (base_type)97.30633};                            \
    vector<base_type> scores = {(base_type)0.2390374, (base_type)0.92039955,   \
                                (base_type)0.05051243, (base_type)0.49574447,  \
                                (base_type)0.8355223};                         \
    int32_t max_output_size = 100;                                             \
    base_type iou_threshold(0.001);                                            \
    base_type score_threshold(0.2);                                            \
    std::unique_ptr<int32_t[]> selected_indices(new int32_t[max_output_size]); \
    int32_t output_expect[2] = {1, 0};                                         \
    vector<void *> datas = {(void *)boxes,                                     \
                            (void *)scores.data(),                             \
                            (void *)&max_output_size,                          \
                            (void *)&iou_threshold,                            \
                            (void *)&score_threshold,                          \
                            (void *)result_summary};                           \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    uint32_t check_ret = CpuKernelAllocatorUtils::CheckOutputDataPtr(0);       \
    EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);                         \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    check_ret =                                                                \
        CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]);        \
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);                                    \
    check_ret =                                                                \
        CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]);        \
    EXPECT_EQ(check_ret, KERNEL_STATUS_OK);                                    \
  }

#define ADD_NULL_CASE(base_type, aicpu_type)                              \
  TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT, TestNonMaxSuppressionV3_null) {  \
    vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32,      \
                                   aicpu_type, aicpu_type, DT_UINT64};    \
    vector<vector<int64_t>> shapes = {{5, 4}, {5}, {1}, {1}, {1}, {-1}};  \
    base_type boxes[] = {                                                 \
        (base_type)24.664701, (base_type)92.11955,  (base_type)6.000731,  \
        (base_type)50.0787,   (base_type)83.716705, (base_type)3.6205719, \
        (base_type)9.723194,  (base_type)46.209377, (base_type)77.11508,  \
        (base_type)74.02599,  (base_type)78.719894, (base_type)31.890612, \
        (base_type)16.0346,   (base_type)21.073357, (base_type)31.530176, \
        (base_type)54.47457,  (base_type)45.646084, (base_type)26.065968, \
        (base_type)30.528221, (base_type)97.30633};                       \
    vector<base_type> scores = {(base_type)0, (base_type)0, (base_type)0, \
                                (base_type)0, (base_type)0};              \
    int32_t max_output_size = 100;                                        \
    base_type iou_threshold(0.001);                                       \
    base_type score_threshold(0.2);                                       \
    vector<void *> datas = {(void *)boxes,                                \
                            (void *)scores.data(),                        \
                            (void *)&max_output_size,                     \
                            (void *)&iou_threshold,                       \
                            (void *)&score_threshold,                     \
                            (void *)result_summary};                      \
    CREATE_NODEDEF(shapes, data_types, datas);                            \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                         \
  }

#define ADD_MEMCOPY_CASE(base_type, aicpu_type)                         \
  TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT,                                \
         TestNonMaxSuppressionV3_memcopy##aicpu_type) {                 \
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64,     \
                                   DT_UINT64};                          \
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};              \
    uint64_t release_flag[2] = {1, 1};                                  \
    uint64_t data_size[2] = {result_summary[1], result_summary[3]};     \
    uint64_t src_pt[2] = {result_summary[0], result_summary[2]};        \
    uint64_t raw_shape_size = result_summary[1];                        \
    uint64_t raw_data_size = result_summary[3];                         \
    void *data_buffer = malloc(raw_data_size);                          \
    void *shape_buffer = malloc(raw_shape_size);                        \
    uint64_t dst_ptr[2] = {reinterpret_cast<uint64_t>(shape_buffer),    \
                           reinterpret_cast<uint64_t>(data_buffer)};    \
    vector<void *> datas = {(void *)release_flag, (void *)data_size,    \
                            (void *)src_pt, (void *)dst_ptr};           \
    CREATE_NODEDEF_MEMCOPY(shapes, data_types, datas);                  \
    RUN_KERNEL(memcpy_node_def, HOST, KERNEL_STATUS_OK);                \
    auto check_ret =                                                    \
        CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[0]); \
    EXPECT_EQ(check_ret, KERNEL_STATUS_PARAM_INVALID);                  \
    check_ret =                                                         \
        CpuKernelAllocatorUtils::CheckOutputDataPtr(result_summary[2]); \
    EXPECT_EQ(raw_shape_size, 8);                                       \
    EXPECT_EQ(raw_data_size, 8);                                        \
    uint64_t raw_shape = *(reinterpret_cast<uint64_t *>(shape_buffer)); \
    EXPECT_EQ(raw_shape, 2);                                            \
    int32_t *raw_data = reinterpret_cast<int32_t *>(data_buffer);       \
    EXPECT_EQ(raw_data[0], 1);                                          \
    EXPECT_EQ(raw_data[1], 0);                                          \
    free(data_buffer);                                                  \
    free(shape_buffer);                                                 \
  }

#define ADD_MEMCOPY_CASE_INVALID_PTR(base_type, aicpu_type)          \
  TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT,                             \
         TestNonMaxSuppressionV3_memcopy_invalid_ptr##aicpu_type) {  \
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64,  \
                                   DT_UINT64};                       \
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};           \
    uint64_t release_flag[2] = {1, 1};                               \
    uint64_t data_size[2] = {0x100, 0x100};                          \
    uint64_t src_pt[2] = {0x80, 0x81};                               \
    uint64_t raw_shape_size = 0x100;                                 \
    uint64_t raw_data_size = 0x100;                                  \
    void *data_buffer = malloc(raw_data_size);                       \
    void *shape_buffer = malloc(raw_shape_size);                     \
    uint64_t dst_ptr[2] = {reinterpret_cast<uint64_t>(shape_buffer), \
                           reinterpret_cast<uint64_t>(data_buffer)}; \
    vector<void *> datas = {(void *)release_flag, (void *)data_size, \
                            (void *)src_pt, (void *)dst_ptr};        \
    CREATE_NODEDEF_MEMCOPY(shapes, data_types, datas);               \
    RUN_KERNEL(memcpy_node_def, HOST, KERNEL_STATUS_INNER_ERROR);    \
    free(data_buffer);                                               \
    free(shape_buffer);                                              \
  }

#define ADD_MEMCOPY_NULL_CASE(base_type, aicpu_type)                 \
  TEST_F(TEST_NON_MAX_SUPPRESSION_V3_UT,                             \
         TestNonMaxSuppressionV3_memcopy_null) {                     \
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64,  \
                                   DT_UINT64};                       \
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}, {2}};           \
    uint64_t release_flag[2] = {1, 1};                               \
    uint64_t data_size[2] = {result_summary[1], result_summary[3]};  \
    uint64_t src_pt[2] = {result_summary[0], result_summary[2]};     \
    uint64_t raw_shape_size = result_summary[1];                     \
    uint64_t raw_data_size = result_summary[3];                      \
    void *data_buffer = nullptr;                                     \
    void *shape_buffer = malloc(raw_shape_size);                     \
    uint64_t dst_ptr[2] = {reinterpret_cast<uint64_t>(shape_buffer), \
                           reinterpret_cast<uint64_t>(data_buffer)}; \
    vector<void *> datas = {(void *)release_flag, (void *)data_size, \
                            (void *)src_pt, (void *)dst_ptr};        \
    CREATE_NODEDEF_MEMCOPY(shapes, data_types, datas);               \
    RUN_KERNEL(memcpy_node_def, HOST, KERNEL_STATUS_OK);             \
    free(shape_buffer);                                              \
  }

ADD_CASE(Eigen::half, DT_FLOAT16)
ADD_MEMCOPY_CASE(Eigen::half, DT_FLOAT16)
ADD_CASE(float, DT_FLOAT)
ADD_MEMCOPY_CASE(float, DT_FLOAT)
ADD_NULL_CASE(float, DT_FLOAT)
ADD_MEMCOPY_NULL_CASE(float, DT_FLOAT)
ADD_MEMCOPY_CASE_INVALID_PTR(float, DT_FLOAT)