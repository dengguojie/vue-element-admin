#include "gtest/gtest.h"
#include "graph/utils/op_desc_utils.h"
#include "split_combination_ops.h"
#include "op_proto_test_util.h"
#include "graph/debug/ge_attr_define.h"

using namespace ge;
using namespace op;

class ConcatDTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "ConcatDTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatDTest TearDown" << std::endl;
  }
};

TEST_F(ConcatDTest, ConcatDInferDataSliceTest_001) {
    ge::op::ConcatD op;
    op.SetAttr("N", 1);
    op.SetAttr("concat_dim", 10);

    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {3, 3}, {100, 200}, {4, 8}, {16, 16}};
    auto format = ge::FORMAT_NDC1HWC0;
    auto ori_format = ge::FORMAT_NDHWC;
    auto x0 =create_desc_shape_range({2, 1, 3, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 1, 100, 5, 48}, ori_format, shape_range);
    op.create_dynamic_input_x(1);
    op.UpdateDynamicInputDesc("x0", 0, x0);

    auto y = create_desc_shape_range({2, 9, 100, 5, 16}, ge::DT_FLOAT16, format, {2, 1, 100, 5, 144}, ori_format, shape_range);
    op.UpdateOutputDesc("y", y);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    std::vector<std::vector<int64_t>> y_data_slice = {{0, 2}, {0, 1}, {0, 9}, {0, 100}, {0, 5}, {0, 16}};
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);

    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}