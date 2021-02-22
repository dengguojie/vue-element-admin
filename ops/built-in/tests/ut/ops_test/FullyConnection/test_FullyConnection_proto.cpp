#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------FullyConnection-------------------
class FullyConnectionProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FullyConnection Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FullyConnection Proto Test TearDown" << std::endl;
  }
};

TEST_F(FullyConnectionProtoTest, fullyConnectionSplicDataTest_1) {
    ge::op::FullyConnection fullyConnection;
    fullyConnection.UpdateInputDesc("x", create_desc_with_ori({4, 1, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{4, 64, 16, 1},ge::FORMAT_NHWC));
    fullyConnection.UpdateInputDesc("w", create_desc_with_ori({1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z,{8, 16, 1, 1},ge::FORMAT_HWCN));
    fullyConnection.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{4, 64, 8, 1},ge::FORMAT_NHWC));
    fullyConnection.SetAttr("num_output", 8);
    fullyConnection.SetAttr("transpose", false);
    fullyConnection.SetAttr("axis", 2);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0,2}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(fullyConnection);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);

    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("w");
    std::vector<std::vector<int64_t>> x_data_slice;
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0,2}, {}, {}};
    std::vector<std::vector<int64_t>> expect_w_data_slice = {{}, {}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(expect_w_data_slice, w_data_slice);
}

TEST_F(FullyConnectionProtoTest, fullyConnectionSplicDataTest_2) {
    ge::op::FullyConnection fullyConnection;
    fullyConnection.UpdateInputDesc("x", create_desc_with_ori({1, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 16, 1, 1},ge::FORMAT_NCHW));
    fullyConnection.UpdateInputDesc("w", create_desc_with_ori({1, 1, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z,{8, 16, 1, 1},ge::FORMAT_NCHW));
    fullyConnection.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 8, 1, 1},ge::FORMAT_NCHW));
    fullyConnection.SetAttr("num_output", 8);
    fullyConnection.SetAttr("transpose", false);
    fullyConnection.SetAttr("axis", 1);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0,3}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(fullyConnection);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);

    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("w");
    std::vector<std::vector<int64_t>> x_data_slice;
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {0,3}, {}, {}};
    std::vector<std::vector<int64_t>> expect_w_data_slice = {{}, {}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(expect_w_data_slice, w_data_slice);
}

TEST_F(FullyConnectionProtoTest, fullyConnectionSplicDataTest_3) {
    ge::op::FullyConnection fullyConnection;
    fullyConnection.UpdateInputDesc("x", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 16, 1, 1},ge::FORMAT_NCHW));
    fullyConnection.UpdateInputDesc("w", create_desc_with_ori({1, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z,{32, 16, 1, 1},ge::FORMAT_NCHW));
    fullyConnection.UpdateOutputDesc("y", create_desc_with_ori({4, 2, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 32, 1, 1},ge::FORMAT_NCHW));
    fullyConnection.SetAttr("num_output", 32);
    fullyConnection.SetAttr("transpose", false);
    fullyConnection.SetAttr("axis", 1);
    std::vector<std::vector<int64_t>> y_data_slice ={{1,3}, {0,1}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(fullyConnection);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);

    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("w");
    std::vector<std::vector<int64_t>> x_data_slice;
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{1,3}, {}, {}, {}, {}};
    std::vector<std::vector<int64_t>> expect_w_data_slice = {{}, {0,1}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(expect_w_data_slice, w_data_slice);

    std::int64_t num_output;
    fullyConnection.GetAttr("num_output", num_output);
    std::int64_t expect_num_output = 16;
    EXPECT_EQ(expect_num_output, num_output);
}