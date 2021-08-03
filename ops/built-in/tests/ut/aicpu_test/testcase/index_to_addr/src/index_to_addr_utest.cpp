#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_INDEX_TO_ADDR_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "IndexToAddr", "IndexToAddr")     \
      .Input({"base_addr", data_types[0], shapes[0], datas[0]})    \
      .Input({"x", data_types[1], shapes[1], datas[1]})            \
      .Output({"addrs_table", data_types[2], shapes[2], datas[2]}) \
      .Attr("ori_shape", ori_shape)                                \
      .Attr("ori_storage_mode", std::string("Matrix"))             \
      .Attr("block_size", block_size)                              \
      .Attr("block_storage_mode", std::string("Matrix"))           \
      .Attr("rank_id", 0)                                          \
      .Attr("dtype", DT_FLOAT)

TEST_F(TEST_INDEX_TO_ADDR_UT, Row0_Col3_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2}, {2}, {4, 4}};
  int64_t input0[2] = {20, 40};
  int64_t input1[2] = {0,3};
  int64_t output[16] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1,
                          (void *)output};
  vector<int64_t> ori_shape = {16, 16};
  vector<int64_t> block_size = {4, 4};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int64_t output_exp[16] = {0, 68, 88, 16, 0, 132, 152,  16,
                            0, 196, 216, 16, 0, 260, 280, 16};
  bool compare = CompareResult(output, output_exp, 16);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_INDEX_TO_ADDR_UT, Row1_Col1_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2}, {2}, {4, 4}};
  int64_t input0[2] = {20, 40};
  int64_t input1[2] = {1,1};
  int64_t output[16] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1,
                          (void *)output};
  vector<int64_t> ori_shape = {16, 16};
  vector<int64_t> block_size = {4, 4};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int64_t output_exp[16] = {0, 292, 312, 16, 0, 356, 376, 16,
                            0, 420, 440, 16, 0, 484, 504, 16};
  bool compare = CompareResult(output, output_exp, 16);
  EXPECT_EQ(compare, true);
}