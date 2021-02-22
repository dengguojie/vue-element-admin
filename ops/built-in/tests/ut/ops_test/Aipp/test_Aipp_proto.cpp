/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_aipp_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "aipp.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class Aipp : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Aipp Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Aipp Proto Test TearDown" << std::endl;
  }
};


TEST_F(Aipp, aipp_data_slice_infer1) {
  ge::op::Aipp op;

  auto tensor_desc = create_desc_with_ori({4,3,224,224}, ge::DT_UINT8, ge::FORMAT_NCHW, {4,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("images", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({4,1,224,224,32}, ge::DT_UINT8, ge::FORMAT_NC1HWC0, {4,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("features", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{0,1}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("features");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_images = op_desc->MutableInputDesc("images");
  std::vector<std::vector<int64_t>> images_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_images, ge::ATTR_NAME_DATA_SLICE, images_data_slice);

  std::vector<std::vector<int64_t>> expected_images_data_slice = {{0,1}, {}, {}, {}};
  EXPECT_EQ(expected_images_data_slice, images_data_slice);
}
