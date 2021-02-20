/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_util_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include "../util/util.h"

class TestUtil : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TestUtil SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TestUtil TearDown" << std::endl;
  }
};

TEST_F(TestUtil, GetNewAxis4NewFormat_NCHW2NC1HWC0) {
  vector<int64_t> new_axis;
  vector<int64_t> expected_new_axis;

  string ori_format = "NCHW";
  string new_format = "NC1HWC0";
  size_t ori_shape_len = 4;
  bool reduce_mode = false;

  reduce_mode = false;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -3, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -4, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 3, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 2, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  reduce_mode = true;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -3, ori_format, new_format, reduce_mode);
  expected_new_axis = {1, 4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -4, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 3, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 2, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {1, 4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);
}

TEST_F(TestUtil, GetNewAxis4NewFormat_NDCHW2NDC1HWC0) {
  vector<int64_t> new_axis;
  vector<int64_t> expected_new_axis;

  string ori_format = "NDCHW";
  string new_format = "NDC1HWC0";
  bool reduce_mode = false;
  size_t ori_shape_len = 5;

  reduce_mode = false;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -3, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -4, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -5, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 4, ori_format, new_format, reduce_mode);
  expected_new_axis = {4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 3, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 2, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  reduce_mode = true;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -3, ori_format, new_format, reduce_mode);
  expected_new_axis = {2, 5};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -4, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -5, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 4, ori_format, new_format, reduce_mode);
  expected_new_axis = {4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 3, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 2, ori_format, new_format, reduce_mode);
  expected_new_axis = {2, 5};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);
}

TEST_F(TestUtil, GetNewAxis4NewFormat_ND2FRACTAL_NZ) {
  vector<int64_t> new_axis;
  vector<int64_t> expected_new_axis;

  string ori_format = "ND";
  string new_format = "FRACTAL_NZ";
  size_t ori_shape_len = 4;
  bool reduce_mode = false;

  reduce_mode = false;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -3, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -4, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 3, ori_format, new_format, reduce_mode);
  expected_new_axis = {2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 2, ori_format, new_format, reduce_mode);
  expected_new_axis = {3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  reduce_mode = true;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {2, 5};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {3, 4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -3, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -4, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 3, ori_format, new_format, reduce_mode);
  expected_new_axis = {2, 5};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 2, ori_format, new_format, reduce_mode);
  expected_new_axis = {3, 4};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);
}

TEST_F(TestUtil, GetNewAxis4NewFormat_ND2FRACTAL_NZ_2) {
  vector<int64_t> new_axis;
  vector<int64_t> expected_new_axis;

  string ori_format = "ND";
  string new_format = "FRACTAL_NZ";
  size_t ori_shape_len = 2;
  bool reduce_mode = false;

  reduce_mode = false;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {0};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {1};
  EXPECT_EQ(new_axis, expected_new_axis);

  reduce_mode = true;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {0, 3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -2, ori_format, new_format, reduce_mode);
  expected_new_axis = {1, 2};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 1, ori_format, new_format, reduce_mode);
  expected_new_axis = {0, 3};
  EXPECT_EQ(new_axis, expected_new_axis);

  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, 0, ori_format, new_format, reduce_mode);
  expected_new_axis = {1, 2};
  EXPECT_EQ(new_axis, expected_new_axis);
}

TEST_F(TestUtil, GetNewAxis4NewFormat_NCHW2FRACTAL_Z) {
  vector<int64_t> new_axis;
  vector<int64_t> expected_new_axis;

  string ori_format = "NCHW";
  string new_format = "FRACTAL_Z"; // C1HWNiNoC0
  size_t ori_shape_len = 4;
  bool reduce_mode = false;

  reduce_mode = false;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {};
  EXPECT_EQ(new_axis, expected_new_axis);
}

TEST_F(TestUtil, GetNewAxis4NewFormat_NCHW2FRACTAL_Z_3D) {
  vector<int64_t> new_axis;
  vector<int64_t> expected_new_axis;

  string ori_format = "NDCHW";
  string new_format = "FRACTAL_Z_3D"; // DC1HWNiNoC0
  size_t ori_shape_len = 5;
  bool reduce_mode = false;

  reduce_mode = false;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {};
  EXPECT_EQ(new_axis, expected_new_axis);
}

TEST_F(TestUtil, GetNewAxis4NewFormat_NHWC2NHWC) {
  vector<int64_t> new_axis;
  vector<int64_t> expected_new_axis;

  string ori_format = "NHWC";
  string new_format = "NHWC"; // DC1HWNiNoC0
  size_t ori_shape_len = 4;
  bool reduce_mode = false;

  reduce_mode = false;
  new_axis = ge::GetNewAxis4NewFormat(ori_shape_len, -1, ori_format, new_format, reduce_mode);
  expected_new_axis = {-1};
  EXPECT_EQ(new_axis, expected_new_axis);
}
