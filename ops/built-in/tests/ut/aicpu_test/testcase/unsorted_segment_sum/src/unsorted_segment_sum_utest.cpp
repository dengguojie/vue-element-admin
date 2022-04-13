/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_UNSORTEDSEGMENTSUM_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                            \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();           \
  NodeDefBuilder(node_def.get(), "UnsortedSegmentSum", "UnsortedSegmentSum") \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})                \
      .Input({"segment_ids", (data_types)[1], (shapes)[1], (datas)[1]})      \
      .Input({"num_segments", (data_types)[2], (shapes)[2], (datas)[2]})     \
      .Output({"y", (data_types)[3], (shapes)[3], (datas)[3]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3, typename T4>
void RunUnsortedSegmentSumKernel(vector<string> data_files,
                                 vector<DataType> &data_types,
                                 vector<vector<int64_t>> &shapes) {
  // read data from file for input1:"x"
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2:"segment_ids"
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  // read data from file for input3:"num_segments"
  data_path = ktestcaseFilePath + data_files[2];
  uint64_t input3_size = CalTotalElements(shapes, 2);
  T3 *input3 = new T3[input3_size];
  status = ReadFile(data_path, input3, input3_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 3);
  T4 *output = new T4[output_size];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[3];
  T4 *output_exp = new T4[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1;
  delete[] input2;
  delete[] input3;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_1.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_1.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_1.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_1.txt"};
  RunUnsortedSegmentSumKernel<int32_t, int32_t, int32_t, int32_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT32_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_2.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_2.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_2.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_2.txt"};
  RunUnsortedSegmentSumKernel<int32_t, int64_t, int64_t, int32_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{5, 6}, {5}, {1}, {5, 6}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_3.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_3.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_3.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_3.txt"};
  RunUnsortedSegmentSumKernel<int16_t, int32_t, int32_t, int16_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT16_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_INT16, DT_INT64, DT_INT64, DT_INT16};
  vector<vector<int64_t>> shapes = {{5, 6}, {5}, {1}, {5, 6}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_4.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_4.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_4.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_4.txt"};
  RunUnsortedSegmentSumKernel<int16_t, int64_t, int64_t, int16_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 3, 3}, {3}, {1}, {3, 3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_5.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_5.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_5.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_5.txt"};
  RunUnsortedSegmentSumKernel<float, int32_t, int32_t, float>(files, data_types,
                                                              shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_FLOAT_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 3, 3}, {3}, {1}, {3, 3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_6.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_6.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_6.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_6.txt"};
  RunUnsortedSegmentSumKernel<float, int64_t, int64_t, float>(files, data_types,
                                                              shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_7.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_7.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_7.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_7.txt"};
  RunUnsortedSegmentSumKernel<double, int32_t, int32_t, double>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_DOUBLE_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_INT64, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_8.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_8.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_8.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_8.txt"};
  RunUnsortedSegmentSumKernel<double, int64_t, int64_t, double>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_9.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_9.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_9.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_9.txt"};
  RunUnsortedSegmentSumKernel<Eigen::half, int32_t, int32_t, Eigen::half>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_FLOAT16_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT64, DT_INT64, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_10.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_10.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_10.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_10.txt"};
  RunUnsortedSegmentSumKernel<Eigen::half, int64_t, int64_t, Eigen::half>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{5, 5, 5}, {5}, {1}, {5, 5, 5}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_11.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_11.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_11.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_11.txt"};
  RunUnsortedSegmentSumKernel<int8_t, int32_t, int32_t, int8_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT8_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_INT8, DT_INT64, DT_INT64, DT_INT8};
  vector<vector<int64_t>> shapes = {{5, 5, 5}, {5}, {1}, {5, 5, 5}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_12.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_12.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_12.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_12.txt"};
  RunUnsortedSegmentSumKernel<int8_t, int64_t, int64_t, int8_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{10, 10}, {10}, {1}, {10, 10}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_13.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_13.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_13.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_13.txt"};
  RunUnsortedSegmentSumKernel<int64_t, int32_t, int32_t, int64_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_INT64_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{10, 10}, {10}, {1}, {10, 10}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_14.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_14.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_14.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_14.txt"};
  RunUnsortedSegmentSumKernel<int64_t, int64_t, int64_t, int64_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{10, 11, 12}, {10}, {1}, {10, 11, 12}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_15.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_15.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_15.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_15.txt"};
  RunUnsortedSegmentSumKernel<uint8_t, int32_t, int32_t, uint8_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT8_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_UINT8, DT_INT64, DT_INT64, DT_UINT8};
  vector<vector<int64_t>> shapes = {{10, 11, 12}, {10}, {1}, {10, 11, 12}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_16.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_16.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_16.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_16.txt"};
  RunUnsortedSegmentSumKernel<uint8_t, int64_t, int64_t, uint8_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_17.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_17.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_17.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_17.txt"};
  RunUnsortedSegmentSumKernel<uint16_t, int32_t, int32_t, uint16_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT16_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_UINT16, DT_INT64, DT_INT64, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_18.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_18.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_18.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_18.txt"};
  RunUnsortedSegmentSumKernel<uint16_t, int64_t, int64_t, uint16_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_19.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_19.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_19.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_19.txt"};
  RunUnsortedSegmentSumKernel<uint32_t, int32_t, int32_t, uint32_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT32_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_UINT32, DT_INT64, DT_INT64, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_20.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_20.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_20.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_20.txt"};
  RunUnsortedSegmentSumKernel<uint32_t, int64_t, int64_t, uint32_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_INT32, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_21.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_21.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_21.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_21.txt"};
  RunUnsortedSegmentSumKernel<uint64_t, int32_t, int32_t, uint64_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_UINT64_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_UINT64, DT_INT64, DT_INT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_22.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_22.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_22.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_22.txt"};
  RunUnsortedSegmentSumKernel<uint64_t, int64_t, int64_t, uint64_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, BIG_DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {
      {4, 9, 1024}, {4}, {1}, {4, 9, 1024}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_23.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_23.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_23.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_23.txt"};
  RunUnsortedSegmentSumKernel<int32_t, int32_t, int32_t, int32_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, BIG_DATA_TYPE_INT32_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{33, 1024}, {33}, {1}, {33, 1024}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_24.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_24.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_24.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_24.txt"};
  RunUnsortedSegmentSumKernel<int32_t, int64_t, int64_t, int32_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, BIG_DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{50, 1500}, {50}, {1}, {50, 1500}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_25.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_25.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_25.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_25.txt"};
  RunUnsortedSegmentSumKernel<int16_t, int32_t, int32_t, int16_t>(
      files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, BIG_DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1024, 33}, {1024}, {1}, {1024, 33}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_26.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_26.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_26.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_26.txt"};
  RunUnsortedSegmentSumKernel<float, int32_t, int32_t, float>(files, data_types,
                                                              shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_INT32,
                                 DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_27.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_27.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_27.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_27.txt"};
  RunUnsortedSegmentSumKernel<std::complex<float>, int32_t, int32_t,
                              std::complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_COMPLEX64_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT64, DT_INT64,
                                 DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_28.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_28.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_28.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_28.txt"};
  RunUnsortedSegmentSumKernel<std::complex<float>, int64_t, int64_t,
                              std::complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_INT32,
                                 DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_29.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_29.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_29.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_29.txt"};
  RunUnsortedSegmentSumKernel<std::complex<double>, int32_t, int32_t,
                              std::complex<double>>(files, data_types, shapes);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, DATA_TYPE_COMPLEX128_SUCC_WITHINT64ID) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_INT64,
                                 DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  vector<string> files{
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input1_30.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input2_30.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_input3_30.txt",
      "unsorted_segment_sum/data/unsorted_segment_sum_data_output1_30.txt"};
  RunUnsortedSegmentSumKernel<std::complex<double>, int64_t, int64_t,
                              std::complex<double>>(files, data_types, shapes);
}
// exception instance
TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, OUTPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 4}};
  int32_t input1[9] = {(int32_t)1};
  int32_t input2[3] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[12] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, INPUT1_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  int32_t input1[9] = {(int32_t)1};
  int32_t input2[3] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int64_t output[9] = {(int64_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  int32_t output[9] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)nullptr,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  int32_t input1[9] = {(int32_t)1};
  int32_t input2[3] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, INPUT1_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  bool input1[9] = {(bool)1};
  int32_t input2[3] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  bool output[9] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, INPUT2_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_BOOL, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  int32_t input1[9] = {(int32_t)1};
  bool input2[3] = {(bool)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[9] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_UNSORTEDSEGMENTSUM_UT, INPUT3_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3}, {1}, {3, 3}};
  int32_t input1[9] = {(int32_t)1};
  int32_t input2[3] = {(int32_t)0};
  bool input3[1] = {(bool)0};
  int32_t output[9] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}