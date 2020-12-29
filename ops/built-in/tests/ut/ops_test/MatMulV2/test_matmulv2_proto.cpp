#include <tuple>

#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "op_proto_test_util.h"

#define OP_TUPLE tuple<vector<int64_t>, DataType, Format, vector<pair<int64_t,int64_t>>>
#define RES_TUPLE tuple<vector<int64_t>, vector<pair<int64_t,int64_t>>>
#define CASE_TUPLE tuple<OP_TUPLE, OP_TUPLE, OP_TUPLE, bool, bool, RES_TUPLE>

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
               RES_TUPLE{{2, 5}, {{2, 2}, {5, 5}}}},
    CASE_TUPLE{OP_TUPLE{{7, 8}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{8, 9}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{2, 17}}}, false, false, RES_TUPLE{{7, 9}, {{7, 7}, {9, 9}}}},
    CASE_TUPLE{OP_TUPLE{{7, 8}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{8, -1}, DT_FLOAT16, FORMAT_ND, {{8, 8}, {4, 33}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{2, 17}}}, false, false, RES_TUPLE{{7, -1}, {{7, 7}, {4, 17}}}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1}, {{1, -1}, {1, 60}}}},
    CASE_TUPLE{OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{5, -1}, {{5, 5}, {1, -1}}}},
    CASE_TUPLE{OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               false,
               false,
               RES_TUPLE{{5, -1}, {{5, 5}, {1, -1}}}},
    // intersect k
    CASE_TUPLE{OP_TUPLE{{7, -1}, DT_FLOAT16, FORMAT_ND, {{7, 7}, {4, 9}}},
               OP_TUPLE{{9, -1}, DT_FLOAT16, FORMAT_ND, {{9, 9}, {7, 12}}},
               {},
               false,
               true,
               RES_TUPLE{{7, 9}, {{7, 7}, {9, 9}}}},
    // all -2
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-2}, {}}},
    /* with bias */
    // intersect k
    CASE_TUPLE{OP_TUPLE{{7, -1}, DT_FLOAT16, FORMAT_ND, {{7, 7}, {4, 9}}},
               OP_TUPLE{{9, -1}, DT_FLOAT16, FORMAT_ND, {{9, 9}, {7, 12}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{4, 80}}}, false, true, RES_TUPLE{{7, 9}, {{7, 7}, {9, 9}}}},
    // intersect n
    CASE_TUPLE{OP_TUPLE{{-1, 5}, DT_FLOAT16, FORMAT_ND, {{4, 9}, {5, 5}}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{4, 80}}}, false, false, RES_TUPLE{{-1, -1}, {{4, 9}, {4, 60}}}},
    // change -1 to fix shape
    CASE_TUPLE{OP_TUPLE{{-1, 5}, DT_FLOAT16, FORMAT_ND, {{1, -1}, {5, 5}}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               OP_TUPLE{{-1}, DT_FLOAT16, FORMAT_ND, {{5, 5}}}, false, false, RES_TUPLE{{-1, 5}, {{1, -1}, {5, 5}}}},
    // all -2
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}}},
    /* dts */
    CASE_TUPLE{OP_TUPLE{{-1, 5}, DT_FLOAT16, FORMAT_ND, {{1, 60}, {5, 5}}},
               OP_TUPLE{{6, -1}, DT_FLOAT16, FORMAT_ND, {{6, 6}, {5, 90}}},
               {},
               true,
               true,
               RES_TUPLE{{5, 6}, {{5, 5}, {6, 6}}}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               true,
               false,
               RES_TUPLE{{-1, -1}, {{1, -1}, {1, 60}}}},
    CASE_TUPLE{OP_TUPLE{{-1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 80}, {1, 80}}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1}, {{1, 80}, {1, 60}}}},
    CASE_TUPLE{OP_TUPLE{{80, 5}, DT_FLOAT16, FORMAT_ND, {{1, 5}, {1, 5}, {16, 16}, {16, 16}}},
               OP_TUPLE{{5, 60}, DT_FLOAT16, FORMAT_ND, {{1, 4}, {1, 1}, {16, 16}, {16, 16}}},
               {},
               false,
               false,
               RES_TUPLE{{80, 60}, {{80, 80}, {60, 60}}}},
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

void Operate(MatMulV2 &op) {
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

void Check(MatMulV2 &op, const RES_TUPLE &expected) {
  auto output_desc = op.GetOutputDesc("y");

  auto shape = output_desc.GetShape().GetDims();
  vector<pair<int64_t,int64_t>> range;
  EXPECT_EQ(output_desc.GetShapeRange(range), GRAPH_SUCCESS);

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

  Operate(op);

  Check(op, get<5>(GetParam()));
}

INSTANTIATE_TEST_CASE_P(DynamicShape,
                        MatMulV2Test,
                        testing::ValuesIn(testcase_matmul));
