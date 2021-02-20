#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------StridedWrite-------------------
class StridedWriteProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StridedWrite Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StridedWrite Proto Test TearDown" << std::endl;
  }
};

// REG_OP(StridedWrite)
//     .INPUT(x, TensorType::ALL())
//     .OUTPUT(y, TensorType::ALL())
//     .ATTR(axis, Int, 1)
//     .ATTR(stride, Int, 1)
//     .OP_END_FACTORY_REG(StridedWrite)

TEST_F(StridedWriteProtoTest, stridedWriteSplicDataTest) {
    ge::op::StridedWrite stdiredwrite;
    stdiredwrite.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    stdiredwrite.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    stdiredwrite.SetAttr("strides", 1);
    stdiredwrite.SetAttr("axis", 1);
    std::vector<std::vector<int64_t>> y_data_slice ={{0,3}, {0,2}, {0,62}, {3,10}, {5,7}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(stdiredwrite);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{0,3}, {0,2}, {0,62}, {3,10}, {5,7}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}