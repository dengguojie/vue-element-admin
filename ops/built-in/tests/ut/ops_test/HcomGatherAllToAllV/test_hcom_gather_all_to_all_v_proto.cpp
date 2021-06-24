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
 * @file test_hcom_gather_all_to_all_v_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "hcom_ops.h"
#include "array_ops.h"
#include "../../../op_proto/util/util.h"

class HcomGatherAllToAllVTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "HcomGatherAllToAllV SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HcomGatherAllToAllV TearDown" << std::endl;
  }
};

TEST_F(HcomGatherAllToAllVTest, hcom_gather_all_to_all_v_infershape_test) {
  ge::op::HcomGatherAllToAllV op;
  op.UpdateInputDesc("addrinfo", create_desc({1, 1}, ge::DT_UINT64));
  op.UpdateInputDesc("addrinfo_count_per_rank", create_desc({1, 1}, ge::DT_INT64));
  op.UpdateInputDesc("send_displacements", create_desc({1, 1}, ge::DT_INT64));
  op.UpdateInputDesc("recv_counts", create_desc({1, 1}, ge::DT_INT64));
  op.UpdateInputDesc("recv_displacements", create_desc({1, 1}, ge::DT_INT64));
  op.SetAttr("group", "hccl_world_group");
  op.SetAttr("dtype", ge::DT_FLOAT);
  op.SetAttr("addr_length", 1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto recv_data_desc = op.get_output_desc_recv_data();
  EXPECT_EQ(recv_data_desc.GetDataType(), ge::DT_FLOAT);
  auto gathered_desc = op.get_output_desc_gathered();
  EXPECT_EQ(gathered_desc.GetDataType(), ge::DT_FLOAT);
}


TEST_F(HcomGatherAllToAllVTest, hcom_gather_all_to_all_v_infershape_test_const) {
  ge::op::HcomGatherAllToAllV op;
  op.UpdateInputDesc("addrinfo_count_per_rank", create_desc({2, 1}, ge::DT_INT64));
  op.UpdateInputDesc("send_displacements", create_desc({2, 1}, ge::DT_INT64));

  {
    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_UINT64);
    constDesc.SetSize(4 * sizeof(int64_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[4] = {1, 1, 2, 1};
    constTensor.SetData((uint8_t*)constData, 4* sizeof(int64_t));
    auto addrinfo = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_addrinfo(addrinfo);
    auto desc = op.GetInputDesc("addrinfo");
    desc.SetDataType(ge::DT_UINT64);
    std::vector<int64_t> dims;
    dims.push_back(4);
    desc.SetShape(ge::Shape(dims));
    op.UpdateInputDesc("addrinfo", desc);
  }

  {
    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT64);
    constDesc.SetSize(2 * sizeof(int64_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[2] = {1, 1};
    constTensor.SetData((uint8_t*)constData, 2* sizeof(int64_t));
    auto recv_counts = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_recv_counts(recv_counts);
    auto desc = op.GetInputDesc("recv_counts");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("recv_counts", desc);
  }

  {
    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT64);
    constDesc.SetSize(2 * sizeof(int64_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[2] = {0, 1};
    constTensor.SetData((uint8_t*)constData, 2* sizeof(int64_t));
    auto recv_displacements = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_recv_displacements(recv_displacements);
    auto desc = op.GetInputDesc("recv_displacements");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("recv_displacements", desc);
  }

  {
    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT64);
    constDesc.SetSize(2 * sizeof(int64_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[2] = {0, 1};
    constTensor.SetData((uint8_t*)constData, 2* sizeof(int64_t));
    auto recv_displacements = ge::op::Constant().set_attr_value(constTensor);
    op.set_input_recv_displacements(recv_displacements);
    auto desc = op.GetInputDesc("recv_displacements");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("recv_displacements", desc);

    vector<uint64_t> addrinfo;
    bool ret = GetConstValue(op, constTensor, ge::DT_UINT64, addrinfo);
  }

  op.SetAttr("group", "hccl_world_group");
  op.SetAttr("dtype", ge::DT_FLOAT);
  op.SetAttr("addr_length", 1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto recv_data_desc = op.get_output_desc_recv_data();
  EXPECT_EQ(recv_data_desc.GetDataType(), ge::DT_FLOAT);
  auto gathered_desc = op.get_output_desc_gathered();
  EXPECT_EQ(gathered_desc.GetDataType(), ge::DT_FLOAT);

}