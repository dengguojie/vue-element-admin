/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

/*!
 * \file test_as_strided_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "all_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"

using namespace std;

class AsStridedTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AsStridedTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AsStridedTiling TearDown" << std::endl;
  }
};

const int64_t profiling_test_num = 0;
static void run_case(std::vector<int64_t> input_shape, std::vector<int64_t> v_size, std::vector<int64_t> v_stride,
                     std::vector<int64_t> v_storage_offset, std::vector<int64_t> output_shape, std::string data_dtype,
                     std::string compile_info, std::string expect_tiling, std::string case_name) {
  using namespace ut_util;

  OP_EVENT("OP_TILING_UTEST", "case_name = %s", case_name.c_str());
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AsStrided");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto test_op = op::AsStrided("AsStrided");

  int64_t size_dims = v_size.size();
  int64_t storage_dims = v_storage_offset.size();
  std::vector<int64_t> size_shape = {size_dims};
  std::vector<int64_t> stride_shape = {size_dims};
  std::vector<int64_t> storage_offset_shape = {storage_dims};

  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_shape, StringToDtype(data_dtype), FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, size, size_shape, DT_INT64, FORMAT_ND, v_size);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, stride, stride_shape, DT_INT64, FORMAT_ND, v_stride);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, storage_offset, storage_offset_shape, DT_INT64, FORMAT_ND, v_storage_offset);
  TENSOR_OUTPUT_WITH_SHAPE(test_op, y, output_shape, StringToDtype(data_dtype), FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  if (expect_tiling != "") {
    EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), expect_tiling);
  }
  for (int64_t i = 0; i < profiling_test_num; i++) {
    RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  }
}

TEST_F(AsStridedTiling, AsStrided_tiling1) {
  std::vector<int64_t> input_shape = {101, 78, 67, 2, 78, 2};
  std::vector<int64_t> output_shape = {3, 2, 66, 691};
  std::vector<int64_t> v_size = {3, 2, 66, 691};
  std::vector<int64_t> v_stride = {11, 32, 31, 1};
  std::vector<int64_t> v_storage_offset = {3};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3000 3 0 4080 691 691 185 0 185 1 0 1 0 1 0 1 26 3 691 1 1 0 691 1 3 132 3 11 66 2 32 1 66 31 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling2) {
  std::vector<int64_t> input_shape = {101, 78, 67, 2, 78, 2};
  std::vector<int64_t> output_shape = {3, 2, 66, 691};
  std::vector<int64_t> v_size = {3, 2, 66, 691};
  std::vector<int64_t> v_stride = {11, 32, 31, 1};
  std::vector<int64_t> v_storage_offset = {3};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3000 5 0 2040 691 691 93 0 93 1 0 1 0 1 0 1 24 3 691 1 1 0 691 1 3 132 3 11 66 2 32 1 66 31 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling3) {
  std::vector<int64_t> input_shape = {101, 78, 67, 2, 78, 2};
  std::vector<int64_t> output_shape = {3, 2, 66, 691};
  std::vector<int64_t> v_size = {3, 2, 66, 691};
  std::vector<int64_t> v_stride = {11, 32, 31, 1};
  std::vector<int64_t> v_storage_offset = {3};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3000 2 0 4064 691 691 370 0 370 1 0 1 0 1 0 1 26 3 691 1 1 0 691 1 3 132 3 11 66 2 32 1 66 31 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling4) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 200, 320, 501};
  std::vector<int64_t> v_size = {1, 200, 320, 501};
  std::vector<int64_t> v_stride = {12992, 224, 200, 2};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3001 32 65280 4080 1001 501 16 0 2000 1 0 1 0 125 0 125 0 1 501 2 1 0 501 1 2 320 200 224 1 320 200 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling5) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 200, 320, 501};
  std::vector<int64_t> v_size = {1, 200, 320, 501};
  std::vector<int64_t> v_stride = {12992, 224, 200, 2};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3001 32 32640 2040 1001 501 16 0 2000 1 0 1 0 125 0 125 0 1 501 2 1 0 501 1 2 320 200 224 1 320 200 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling6) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2, 1, 50001};
  std::vector<int64_t> v_size = {1, 2, 1, 50001};
  std::vector<int64_t> v_stride = {12992, 224, 200, 2};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3001 25 130560 4064 4031 2016 2 1 2016 1 0 1 1617 1 0 1 0 1 50001 2 1 0 50001 1 1 1 2 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling7) {
  std::vector<int64_t> input_shape = {16, 320, 624, 128};
  std::vector<int64_t> output_shape = {3, 4, 3, 2, 2, 2, 3};
  std::vector<int64_t> v_size = {3, 4, 3, 2, 2, 2, 3};
  std::vector<int64_t> v_stride = {5, 6, 4, 3, 2, 3, 3};
  std::vector<int64_t> v_storage_offset = {7};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3002 3 130560 4064 7 3 127 0 127 1 0 1 0 1 0 1 34 7 3 3 1 0 3 1 6 96 3 5 24 4 6 8 3 4 4 2 3 2 2 2 1 2 3 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling8) {
  std::vector<int64_t> input_shape = {16, 320, 624, 128};
  std::vector<int64_t> output_shape = {3, 4, 3, 2, 2, 2, 3};
  std::vector<int64_t> v_size = {3, 4, 3, 2, 2, 2, 3};
  std::vector<int64_t> v_stride = {5, 6, 4, 3, 2, 3, 3};
  std::vector<int64_t> v_storage_offset = {7};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3002 2 65280 4080 7 3 255 0 255 1 0 1 0 1 0 1 33 7 3 3 1 0 3 1 6 96 3 5 24 4 6 8 3 4 4 2 3 2 2 2 1 2 3 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling9) {
  std::vector<int64_t> input_shape = {16, 320, 624, 128};
  std::vector<int64_t> output_shape = {3, 4, 3, 2, 2, 2, 3};
  std::vector<int64_t> v_size = {3, 4, 3, 2, 2, 2, 3};
  std::vector<int64_t> v_stride = {5, 6, 4, 3, 2, 3, 3};
  std::vector<int64_t> v_storage_offset = {7};
  std::string dtype = "int64";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 32640, \"core_num\": 32}}";
  std::string expect_tiling =
      "3002 3 16320 1020 7 3 127 0 127 1 0 1 0 1 0 1 34 7 3 3 1 0 3 1 6 96 3 5 24 4 6 8 3 4 4 2 3 2 2 2 1 2 3 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling10) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 20, 32, 501};
  std::vector<int64_t> v_size = {1, 20, 32, 501};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3003 32 63240 2040 10458 501 4 0 20 1 0 1 0 5 0 5 0 1 501 0 1 0 501 1 2 32 20 224 1 32 200 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling11) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2, 1, 10001};
  std::vector<int64_t> v_size = {1, 2, 1, 10001};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3003 3 257056 4064 226 4064 1 1 4064 1 0 1 1873 2 0 2 0 1 10001 0 1 0 10001 1 1 1 2 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling12) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 32, 501};
  std::vector<int64_t> v_size = {1, 2000, 32, 501};
  std::vector<int64_t> v_stride = {12992, 5220, 200, 300};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3004 32 32640 2040 1 248 16 0 2000 3 5 3 5 125 0 125 0 1 501 300 1 0 501 1 2 32 2000 5220 1 32 200 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling13) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2, 1, 50001};
  std::vector<int64_t> v_size = {1, 2, 1, 50001};
  std::vector<int64_t> v_stride = {12992, 5220, 200, 300};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3004 31 130560 4064 1 96 2 1 1632 17 0 11 81 1 0 1 0 1 50001 300 1 0 50001 1 1 1 2 5220 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling14) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 32, 5};
  std::vector<int64_t> v_size = {1, 2000, 32, 5};
  std::vector<int64_t> v_stride = {12992, 5220, 200, 300};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3005 32 32640 2040 1 5 49 0 2009 1 0 1 0 41 0 36 6 1 5 300 1 0 5 1 2 32 2000 5220 1 32 200 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling15) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 32, 5};
  std::vector<int64_t> v_size = {1, 2000, 32, 5};
  std::vector<int64_t> v_stride = {12992, 5220, 200, 300};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3005 32 130560 4064 1 5 19 0 2014 1 0 1 0 106 0 83 8 1 5 300 1 0 5 1 2 32 2000 5220 1 32 200 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling16) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 32, 5};
  std::vector<int64_t> v_size = {1, 2000, 32, 5};
  std::vector<int64_t> v_stride = {12992, 224, 2, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3006 32 32640 2040 63 160 16 0 64 1 0 1 0 4 0 1 0 1 5 0 32 2 160 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling17) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 32, 5};
  std::vector<int64_t> v_size = {1, 2000, 32, 5};
  std::vector<int64_t> v_stride = {12992, 224, 2, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3006 32 130560 4064 63 160 16 0 64 1 0 1 0 4 0 1 0 1 5 0 32 2 160 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling18) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 320, 501};
  std::vector<int64_t> v_size = {1, 2000, 320, 501};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3007 32 65280 4080 63801 320 1 0 63 1 0 1 0 63 0 47 0 1 501 0 320 200 160320 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling19) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 320, 501};
  std::vector<int64_t> v_size = {1, 2000, 320, 501};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3007 32 32640 2040 32601 164 1 0 63 2 156 2 156 63 0 47 0 1 501 0 320 200 160320 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling20) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 320, 501};
  std::vector<int64_t> v_size = {1, 2000, 320, 501};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3007 32 130560 4064 63801 320 1 0 63 1 0 1 0 63 0 47 0 1 501 0 320 200 160320 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling21) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 320, 21};
  std::vector<int64_t> v_size = {1, 2000, 320, 21};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3008 32 65280 4080 4001 21 16 0 64 16 5 16 5 4 0 1 0 1 21 0 320 200 6720 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling22) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 200, 320, 7};
  std::vector<int64_t> v_size = {1, 200, 320, 7};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3008 30 32640 2040 2001 11 16 1 11 1 0 1 1 13 8 13 8 1 7 0 320 200 2240 1 1 1 200 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling23) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 320, 15};
  std::vector<int64_t> v_size = {1, 2000, 320, 15};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int64";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 32640, \"core_num\": 32}}";
  std::string expect_tiling =
      "3008 32 16320 1020 1001 6 16 0 64 54 2 54 2 4 0 1 0 1 15 0 320 200 4800 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling24) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {1, 2000, 320, 21};
  std::vector<int64_t> v_size = {1, 2000, 320, 21};
  std::vector<int64_t> v_stride = {12992, 224, 200, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3008 32 130560 4064 4001 21 16 0 64 16 5 16 5 4 0 1 0 1 21 0 320 200 6720 1 1 1 2000 224 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling25) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {256, 1280, 3};
  std::vector<int64_t> v_size = {256, 1280, 3};
  std::vector<int64_t> v_stride = {1, 256, 0};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3009 15 65280 4080 240 240 256 0 256 2 16 2 16 1 0 1 0 1 256 1 1 0 3840 16 2 3 1280 256 1 3 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling26) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {20, 201, 321, 21};
  std::vector<int64_t> v_size = {20, 201, 321, 21};
  std::vector<int64_t> v_stride = {52, 224, 200, 1992};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float32";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 65280, \"core_num\": 32}}";
  std::string expect_tiling =
      "3009 32 32640 2040 209 5 128 0 42368 4 0 4 0 331 0 325 61 1 20 52 1 0 1354941 8 3 6741 201 224 21 321 200 1 21 1992 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling27) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {20, 201, 321, 21};
  std::vector<int64_t> v_size = {20, 201, 321, 21};
  std::vector<int64_t> v_stride = {52, 224, 200, 1992};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int64";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 32640, \"core_num\": 32}}";
  std::string expect_tiling =
      "3009 32 16320 1020 209 5 64 0 42368 4 0 4 0 662 0 649 61 1 20 52 1 0 1354941 4 3 6741 201 224 21 321 200 1 21 1992 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling28) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {20, 201, 321, 21};
  std::vector<int64_t> v_size = {20, 201, 321, 21};
  std::vector<int64_t> v_stride = {52, 224, 2000, 1992};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3009 32 130560 4064 53 2 512 0 42496 10 0 10 0 83 0 74 189 1 20 52 1 0 1354941 32 3 6741 201 224 21 321 2000 1 21 1992 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling29) {
  std::vector<int64_t> input_shape = {146, 43, 142, 344};
  std::vector<int64_t> output_shape = {5893, 640};
  std::vector<int64_t> v_size = {5893, 640};
  std::vector<int64_t> v_stride = {1, 11786};
  std::vector<int64_t> v_storage_offset = {1};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3009 25 65280 4080 240 240 256 1 240 1 0 1 133 3 128 3 128 1 5893 1 1 0 640 16 1 1 640 11786 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling30) {
  std::vector<int64_t> input_shape = {16, 320, 624, 128};
  std::vector<int64_t> output_shape = {3, 1, 2, 2, 3, 3, 2};
  std::vector<int64_t> v_size = {3, 1, 2, 2, 3, 3, 2};
  std::vector<int64_t> v_stride = {5, 0, 0, 18, 6, 2, 1};
  std::vector<int64_t> v_storage_offset = {7};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3002 1 130560 4064 36 36 6 0 6 1 0 1 0 1 0 1 0 7 36 1 1 0 36 1 2 2 3 5 1 2 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling31) {
  std::vector<int64_t> input_shape = {16, 320, 624, 128};
  std::vector<int64_t> output_shape = {3, 4, 3, 2, 3, 2, 3};
  std::vector<int64_t> v_size = {3, 4, 3, 2, 3, 2, 3};
  std::vector<int64_t> v_stride = {5, 0, 0, 1, 0, 0, 1};
  std::vector<int64_t> v_storage_offset = {7};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3002 4 130560 4064 3 3 127 0 127 1 0 1 0 1 0 1 51 7 3 1 1 0 3 1 4 144 3 5 12 12 0 6 2 1 1 6 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling32) {
  std::vector<int64_t> input_shape = {16, 320, 624, 128};
  std::vector<int64_t> output_shape = {3, 4, 1, 2, 3, 1, 1};
  std::vector<int64_t> v_size = {3, 4, 1, 2, 3, 1, 1};
  std::vector<int64_t> v_stride = {5, 0, 0, 1, 0, 0, 1};
  std::vector<int64_t> v_storage_offset = {7};
  std::string dtype = "int8";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 261120, \"core_num\": 32}}";
  std::string expect_tiling =
      "3003 1 257056 4064 19 3 24 0 24 1 0 1 0 1 0 1 0 7 3 0 1 0 3 1 3 8 3 5 2 4 0 1 2 1 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(AsStridedTiling, AsStrided_tiling33) {
  std::vector<int64_t> input_shape = {8, 2268, 21};
  std::vector<int64_t> output_shape = {8, 126, 3, 3};
  std::vector<int64_t> v_size = {8, 126, 3, 3};
  std::vector<int64_t> v_stride = {47628, 1, 378, 126};
  std::vector<int64_t> v_storage_offset = {0};
  std::string dtype = "float16";

  std::string compile_info =
      "{\"vars\": {\"max_elem_cnt\": 130560, \"core_num\": 32}}";
  std::string expect_tiling =
      "3006 1 65280 4080 1134 1134 8 0 8 1 0 1 0 1 0 1 0 0 9 126 126 1 1134 1 1 1 8 47628 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input_shape, v_size, v_stride, v_storage_offset, output_shape, dtype, compile_info, expect_tiling,
           this->test_info_->name());
}
