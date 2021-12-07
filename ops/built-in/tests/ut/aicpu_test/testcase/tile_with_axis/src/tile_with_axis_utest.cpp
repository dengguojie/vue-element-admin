/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 */

#include <gtest/gtest.h>
#ifndef private
#define private public
#define protected public
#endif
#include <Eigen/Core>
#include <iostream>

#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_TILEWITHAXIS_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, axis, tiles)     \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "TileWithAxis", "TileWithAxis")   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})           \
      .Attr("axis", axis)                                          \
      .Attr("tiles", tiles);

template <typename T>
void RunTestTileWithAxis(string test_case_no, vector<DataType> data_types, int32_t axis, int32_t tiles) {
  // 2. 获取input shapes
  vector<int64_t> inputshape_data;
  string input_shape_path =
      ktestcaseFilePath + "tile_with_axis/data/tilewithaxis_data_input_shape_" + test_case_no + ".txt";
  EXPECT_EQ(ReadFile(input_shape_path, inputshape_data), true);

  // 3. 获取output shapes
  uint64_t dims = inputshape_data.size();
  axis = (axis < 0) ? (axis + dims) : axis;
  vector<int64_t> outputshape_data;
  outputshape_data = inputshape_data;
  outputshape_data[axis] = inputshape_data[axis] * tiles;

  // 4. 获取input数据
  uint64_t inputdata_size = 1;
  for (uint64_t i = 0; i < dims; i++) {
    inputdata_size *= inputshape_data[i];
  }
  T input_data[inputdata_size] = {0};
  string input_data_path = ktestcaseFilePath + "tile_with_axis/data/tilewithaxis_input_data_" + test_case_no + ".txt";
  EXPECT_EQ(ReadFile(input_data_path, input_data, inputdata_size), true);

  // 5. 获取outout数据 暂时为NULL
  uint64_t outputdata_size = 1;
  for (uint64_t i = 0; i < dims; i++) {
    outputdata_size *= outputshape_data[i];
  }
  T output_data[outputdata_size] = {0};

  // 6. 创建node，运行算子
  vector<vector<int64_t>> shapes = {inputshape_data, outputshape_data};
  vector<void*> datas = {(void*)input_data, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas, axis, tiles);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK)

  // 7. 对比结果
  string output_data_path = ktestcaseFilePath + "tile_with_axis/data/tilewithaxis_output_data_" + test_case_no + ".txt";
  T output_exp[outputdata_size] = {0};
  EXPECT_EQ(ReadFile(output_data_path, output_exp, outputdata_size), true);

  bool compare = CompareResult(output_data, output_exp, outputdata_size);
  EXPECT_EQ(compare, true);
}

// axis正数，各参数正确， int64数据类型, 3维数据，对应tile_with_gen_data.py中的第1组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT64_SUCC1) {
  RunTestTileWithAxis<int64_t>("1", {DT_INT64, DT_INT64}, 1, 2);
}

// axis正数，各参数正确，int32数据类型, 3维数据，对应tile_with_gen_data.py中的第2组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT32_SUCC2) {
  RunTestTileWithAxis<int32_t>("2", {DT_INT32, DT_INT32}, 1, 2);
}

// axis正数，各参数正确，int16数据类型, 3维数据，对应tile_with_gen_data.py中的第3组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT16_SUCC3) {
  RunTestTileWithAxis<int16_t>("3", {DT_INT16, DT_INT16}, 0, 2);
}

// axis正数，各参数正确，int8数据类型, 3维数据，对应tile_with_gen_data.py中的第4组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT8_SUCC4) {
  RunTestTileWithAxis<int8_t>("4", {DT_INT8, DT_INT8}, 2, 2);
}

// axis正数，各参数正确， uint64数据类型，3维数据，对应tile_with_gen_data.py中的第5组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_UINT64_SUCC5) {
  RunTestTileWithAxis<uint64_t>("5", {DT_UINT64, DT_UINT64}, 1, 2);
}

// axis正数，各参数正确，uint32数据类型，3维数据，对应tile_with_gen_data.py中的第6组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_UINT32_SUCC6) {
  RunTestTileWithAxis<uint32_t>("6", {DT_UINT32, DT_UINT32}, 1, 2);
}

// axis正数，各参数正确，uint16数据类型，3维数据，对应tile_with_gen_data.py中的第7组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_UINT16_SUCC7) {
  RunTestTileWithAxis<uint16_t>("7", {DT_UINT16, DT_UINT16}, 0, 2);
}

// axis正数，各参数正确，uint8数据类型，3维数据，对应tile_with_gen_data.py中的第8组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_UINT8_SUCC8) {
  RunTestTileWithAxis<uint8_t>("8", {DT_UINT8, DT_UINT8}, 2, 2);
}

// axis正数，各参数正确，float16数据类型，3维数据，对应tile_with_gen_data.py中的第9组数据
/* TEST_F(TEST_TILEWITHAXIS_UT, INPUT_FLOAT_SUCC9) {
    RunTestTileWithAxis<Eigen::half>("9", {DT_FLOAT16, DT_FLOAT16}, 1, 2);
} */

// axis正数，各参数正确，float数据类型，3维数据，对应tile_with_gen_data.py中的第10组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_FLOAT_SUCC10) {
  RunTestTileWithAxis<float>("10", {DT_FLOAT, DT_FLOAT}, 1, 2);
}

// axis正数，各参数正确，4维数据，对应tile_with_gen_data.py中的11组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT64_SUCC11) {
  RunTestTileWithAxis<int64_t>("11", {DT_INT64, DT_INT64}, 1, 2);
}

// axis负数，axis=-3和axis=1结果相同， 各参数正确，4维数据，对应tile_with_gen_data.py中的11组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT64_SUCC11_1) {
  RunTestTileWithAxis<int64_t>("11", {DT_INT64, DT_INT64}, -3, 2);
}

// axis正数，各参数正确，5维数据，对应tile_with_gen_data.py中的12组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT32_SUCC12) {
  RunTestTileWithAxis<int32_t>("12", {DT_INT32, DT_INT32}, 1, 2);
}

// axis正数，各参数正确，6维数据，对应tile_with_gen_data.py中的13组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT16_SUCC13) {
  RunTestTileWithAxis<int16_t>("13", {DT_INT16, DT_INT16}, 1, 2);
}

// axis正数，各参数正确，7维数据，对应tile_with_gen_data.py中的14组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT8_SUCC14) {
  RunTestTileWithAxis<int8_t>("14", {DT_INT8, DT_INT8}, 6, 2);
}

// axis正数，各参数正确，8维数据，对应tile_with_gen_data.py中的15组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT8_SUCC15) {
  RunTestTileWithAxis<int8_t>("15", {DT_INT8, DT_INT8}, 6, 2);
}

// axis正数，各参数正确，2维数据，对应tile_with_gen_data.py中的12组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT32_SUCC16) {
  RunTestTileWithAxis<int32_t>("16", {DT_INT32, DT_INT32}, 0, 2);
}

// axis正数，各参数正确，1维数据，对应tile_with_gen_data.py中的12组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT32_SUCC17) {
  RunTestTileWithAxis<int32_t>("17", {DT_INT32, DT_INT32}, 0, 2);
}

// tile = 1 各参数正确，4维数据，对应tile_with_gen_data.py中的11组数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT32_SUCC18) {
  RunTestTileWithAxis<int32_t>("18", {DT_INT32, DT_INT32}, 0, 1);
}

// axis = 4，正数超过最大范围，int64数据类型, 2维数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT64_FAIL1) {
  vector<vector<int64_t>> shapes = {{1, 2}, {2, 2}};
  int64_t input_data[1 * 2] = {0};
  int64_t output_data[2 * 2] = {0};
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<void*> datas = {(void*)input_data, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas, 4, 2);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// axis = -4，正数超过最大范围，int64数据类型, 2维数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT64_FAIL2) {
  vector<vector<int64_t>> shapes = {{1, 2}, {2, 2}};
  int64_t input_data[1 * 2] = {0};
  int64_t output_data[2 * 2] = {0};
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<void*> datas = {(void*)input_data, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas, -4, 2);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// tile = -2，为负数，int64数据类型, 2维数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT64_FAIL3) {
  vector<vector<int64_t>> shapes = {{1, 2}, {2, 2}};
  int64_t input_data[1 * 2] = {0};
  int64_t output_data[2 * 2] = {0};
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<void*> datas = {(void*)input_data, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas, 0, -2);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// input[axis] * tile != ouput[axis]，int64数据类型, 2维数据
TEST_F(TEST_TILEWITHAXIS_UT, INPUT_INT64_FAIL4) {
  vector<vector<int64_t>> shapes = {{1, 2}, {4, 2}};
  int64_t input_data[1 * 2] = {0};
  int64_t output_data[4 * 2] = {0};
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<void*> datas = {(void*)input_data, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas, 1, 2);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}