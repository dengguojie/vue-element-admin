#include <tuple>

#include "common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#define OP_TUPLE tuple<vector<int64_t>, DataType, Format, vector<pair<int64_t,int64_t>>>
#define RES_TUPLE tuple<vector<int64_t>, vector<pair<int64_t,int64_t>>, bool>
#define CASE_TUPLE tuple<OP_TUPLE, OP_TUPLE, OP_TUPLE, bool, bool, RES_TUPLE>
#define PASS true
#define FAILED false

using namespace std;
using namespace ge;
using namespace op;

vector<CASE_TUPLE> testcase_matmul = {
    /* no bias */
    CASE_TUPLE{OP_TUPLE{{2, 4}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{2, 5}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{7, 8}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{8, 9}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{2, 17}}}, false, false, RES_TUPLE{{7, 9}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{7, 8}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{8, -1}, DT_FLOAT16, FORMAT_ND, {{8, 8}, {4, 33}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{2, 17}}}, false, false, RES_TUPLE{{7, -1}, {{7, 7}, {4, 17}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1}, {{1, -1}, {1, 60}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{5, -1}, {{5, 5}, {1, -1}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               false,
               false,
               RES_TUPLE{{5, -1}, {{5, 5}, {1, -1}}, PASS}},
    // intersect k
    CASE_TUPLE{OP_TUPLE{{7, -1}, DT_FLOAT16, FORMAT_ND, {{7, 7}, {4, 9}}},
               OP_TUPLE{{9, -1}, DT_FLOAT16, FORMAT_ND, {{9, 9}, {7, 12}}},
               {},
               false,
               true,
               RES_TUPLE{{7, 9}, {}, PASS}},
    // all -2
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-2}, {}, PASS}},
    /* with bias */
    // intersect k
    CASE_TUPLE{OP_TUPLE{{7, -1}, DT_FLOAT16, FORMAT_ND, {{7, 7}, {4, 9}}},
               OP_TUPLE{{9, -1}, DT_FLOAT16, FORMAT_ND, {{9, 9}, {7, 12}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{4, 80}}}, false, true, RES_TUPLE{{7, 9}, {}, PASS}},
    // intersect n
    CASE_TUPLE{OP_TUPLE{{-1, 5}, DT_FLOAT16, FORMAT_ND, {{4, 9}, {5, 5}}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{4, 80}}}, false, false, RES_TUPLE{{-1, -1}, {{4, 9}, {4, 60}}, PASS}},
    // change -1 to fix shape
    CASE_TUPLE{OP_TUPLE{{-1, 5}, DT_FLOAT16, FORMAT_ND, {{1, -1}, {5, 5}}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{5, 5}}}, false, false, RES_TUPLE{{-1, 5}, {{1, -1}, {5, 5}}, PASS}},
    // all -2
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}, PASS}},
    /* dts */
    CASE_TUPLE{OP_TUPLE{{-1, 5}, DT_FLOAT16, FORMAT_ND, {{1, 60}, {5, 5}}},
               OP_TUPLE{{6, -1}, DT_FLOAT16, FORMAT_ND, {{6, 6}, {5, 90}}},
               {},
               true,
               true,
               RES_TUPLE{{5, 6}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               true,
               false,
               RES_TUPLE{{-1, -1}, {{1, -1}, {1, 60}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 80}, {1, 80}}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1}, {{1, 80}, {1, 60}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{80, 5}, DT_FLOAT16, FORMAT_ND, {{1, 5}, {1, 5}, {16, 16}, {16, 16}}},
               OP_TUPLE{{5, 60}, DT_FLOAT16, FORMAT_ND, {{1, 4}, {1, 1}, {16, 16}, {16, 16}}},
               {},
               false,
               false,
               RES_TUPLE{{80, 60}, {}, PASS}},
    /* dfx */
    // fix shape: Don't care about range
    //   case 1: empty range
    CASE_TUPLE{OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{1, 1}, {}, PASS}},
    //    case 2: error range
    CASE_TUPLE{OP_TUPLE{{2, 1}, DT_FLOAT16, FORMAT_ND, {{2, 1}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{2, 1}, {}, PASS}},
    //    case 3: error range
    CASE_TUPLE{OP_TUPLE{{2, 1}, DT_FLOAT16, FORMAT_ND, {{2, 1}, {2, 2}, {3, 3}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{2, 1}, {}, PASS}},
    // dynamic shape:
    //    case 1: empty range with all -1 in shape
    CASE_TUPLE{OP_TUPLE{{-1, -1}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{1, -1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1}, {{1, -1}, {1, -1}}, PASS}},
    //    case 2: empty range with a few -1 in shape
    CASE_TUPLE{OP_TUPLE{{-1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-1, 1}, {{1, -1}, {1, 1}}, PASS}},
    //    case 3: range is correct
    CASE_TUPLE{OP_TUPLE{{-1, 1}, DT_FLOAT16, FORMAT_ND, {{3, 4}, {1, 1}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-1, 1}, {{3, 4}, {1, 1}}, PASS}},
    //    case 4: range is error(dim error)
    CASE_TUPLE{OP_TUPLE{{-1, 1}, DT_FLOAT16, FORMAT_ND, {{3, 4}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{}, {}, FAILED}},
    //    case 5: range is error(dim error)
    CASE_TUPLE{OP_TUPLE{{-1, 1}, DT_FLOAT16, FORMAT_ND, {{3, 4}, {2, 3}, {3, 3}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{}, {}, FAILED}},
    //    case 6: range is error(range error)
    CASE_TUPLE{OP_TUPLE{{-1, 1}, DT_FLOAT16, FORMAT_ND, {{6, 3}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{}, {}, FAILED}},
    //    case 7: range is error(range error)
    CASE_TUPLE{OP_TUPLE{{-1, 1}, DT_FLOAT16, FORMAT_ND, {{6, 3}, {1, 1}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{}, {}, FAILED}},
    //    case 8: range is error(range error)
    CASE_TUPLE{OP_TUPLE{{-1, 1}, DT_FLOAT16, FORMAT_ND, {{6, 3}, {}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{}, {}, FAILED}},
    // unkown rank: Don't care about range
    //    case 1: empty range
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {{}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-1, 1}, {{1, -1}, {1, 1}}, PASS}},
    //    case 2: with range
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {{3, 3}}},
               OP_TUPLE{{1, 1}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-1, 1}, {{1, -1}, {1, 1}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{32, 32}, DT_FLOAT16, FORMAT_ND, {}},
            OP_TUPLE{{-1, 64}, DT_FLOAT16, FORMAT_ND, {{4,5}}},
            {},
            false,
            false,
            RES_TUPLE{{}, {}, FAILED}},
    CASE_TUPLE{OP_TUPLE{{32, -1}, DT_FLOAT16, FORMAT_ND, {{2, 3}}},
            OP_TUPLE{{-1, 64}, DT_FLOAT16, FORMAT_ND, {{4,5}}},
            {},
            false,
            false,
            RES_TUPLE{{}, {}, FAILED}},
    CASE_TUPLE{OP_TUPLE{{32, -1}, DT_FLOAT16, FORMAT_ND, {{2, 3}}},
            OP_TUPLE{{64, 64}, DT_FLOAT16, FORMAT_ND, {}},
            {},
            false,
            false,
            RES_TUPLE{{}, {}, FAILED}},
    CASE_TUPLE{OP_TUPLE{{32, 32}, DT_FLOAT16, FORMAT_ND, {}},
            OP_TUPLE{{-1, 64}, DT_FLOAT16, FORMAT_ND, {{4, 5}}},
            {},
            false,
            false,
            RES_TUPLE{{}, {}, FAILED}},
    CASE_TUPLE{OP_TUPLE{{-1, -1}, DT_FLOAT16, FORMAT_ND, {}},
            OP_TUPLE{{122, 128}, DT_FLOAT16, FORMAT_ND, {{122, 128}}},
            {},
            false,
            false,
            RES_TUPLE{{-1, 128}, {{1, -1}, {128, 128}}, PASS}},
    // support vector op
    CASE_TUPLE{OP_TUPLE{{32, 32}, DT_FLOAT16, FORMAT_ND, {}},
            OP_TUPLE{{32, 64}, DT_FLOAT16, FORMAT_ND, {{4, 5}}},
            {},
            false,
            false,
            RES_TUPLE{{32, 64}, {}, PASS}},
};

MatMulV2 CreateMatMulV2Op(OP_TUPLE a, OP_TUPLE b, OP_TUPLE bias,
                          bool trans_a, bool trans_b) {
  auto tensor_a = create_desc_shape_range(get<0>(a), get<1>(a), get<2>(a), get<0>(a), get<2>(a), get<3>(a));
  auto tensor_b = create_desc_shape_range(get<0>(b), get<1>(b), get<2>(b), get<0>(b), get<2>(b), get<3>(b));
  TensorDesc tensor_bias;
  auto shape_bias = get<0>(bias);
  if (!shape_bias.empty()) {
    tensor_bias = create_desc_shape_range(get<0>(bias), get<1>(bias), get<2>(bias), get<0>(bias), get<2>(bias), get<3>(bias));
  }

  MatMulV2 op;
  op.UpdateInputDesc("x1", tensor_a);
  op.UpdateInputDesc("x2", tensor_b);
  if (!shape_bias.empty()) {
    op.UpdateInputDesc("bias", tensor_bias);
  }
  op.SetAttr("transpose_x1", trans_a);
  op.SetAttr("transpose_x2", trans_b);
  return op;
}

void Operate(MatMulV2& op, bool expected_result) {
  auto ret = op.InferShapeAndType();

  if (expected_result == PASS) {
    EXPECT_EQ(ret, GRAPH_SUCCESS);
  } else {
    EXPECT_EQ(ret, GRAPH_FAILED);
  }
}

void Check(MatMulV2 &op, const RES_TUPLE &expected) {
  auto output_desc = op.GetOutputDesc("y");

  auto shape = output_desc.GetShape().GetDims();
  vector<pair<int64_t,int64_t>> range;
  EXPECT_EQ(output_desc.GetShapeRange(range), GRAPH_SUCCESS);

  cout << "shape of output: (";
  for (auto v : shape) {
    cout << v << ", ";
  }
  cout << ")" << endl;
  cout << "range of output: (";
  for (auto v : range) {
    cout << "(" << v.first << ", " << v.second << "), ";
  }
  cout << ")" << endl;

  EXPECT_EQ(shape, get<0>(expected));
  EXPECT_EQ(range, get<1>(expected));
}

class MatMulV2Test :
  public testing::TestWithParam<CASE_TUPLE> {
public:
  MatMulV2Test() {}
};

TEST_P(MatMulV2Test, General) {
  auto op = CreateMatMulV2Op(std::get<0>(GetParam()),
                             std::get<1>(GetParam()),
                             std::get<2>(GetParam()),
                             std::get<3>(GetParam()),
                             std::get<4>(GetParam()));

  auto result_tuple = get<5>(GetParam());
  auto expected_result = get<2>(result_tuple);

  Operate(op, expected_result);

  if (expected_result == PASS) {
    Check(op, result_tuple);
  }
}

INSTANTIATE_TEST_CASE_P(DynamicShape,
                        MatMulV2Test,
                        testing::ValuesIn(testcase_matmul));

// cut n in NZ
TEST_F(MatMulV2Test, split_test0) {
  ge::op::MatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{0, 1}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);

  std::vector<std::vector<int64_t>> expect_x2_data_slice = {{0, 1}, {}, {}, {}};
  EXPECT_EQ(expect_x2_data_slice, x2_data_slice);
}

// cut m in NZ
TEST_F(MatMulV2Test, split_test1) {
  ge::op::MatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 1}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{}, {0, 1}, {}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);
}

// cut n in ND
TEST_F(MatMulV2Test, split_test2) {
  ge::op::MatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 15}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);

  std::vector<std::vector<int64_t>> expect_x2_data_slice = {{}, {0, 15}};
  EXPECT_EQ(expect_x2_data_slice, x2_data_slice);
}

// cut m in ND
TEST_F(MatMulV2Test, split_test3) {
  ge::op::MatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{16, 31}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{16, 31}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);
}


TEST_F(MatMulV2Test, MatMulV2InferShapeTest) {
  ge::op::MatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}