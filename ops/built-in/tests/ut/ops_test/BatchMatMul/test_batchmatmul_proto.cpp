#include <tuple>

#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "op_proto_test_util.h"

using namespace std;
using namespace ge;
using namespace op;

#define OP_TUPLE tuple<vector<int64_t>, DataType, Format, vector<pair<int64_t,int64_t>>>

BatchMatMul CreateBatchMatMulOp(OP_TUPLE a, OP_TUPLE b,
                                bool trans_a, bool trans_b) {
  auto tensor_a = create_desc_shape_range(get<0>(a), get<1>(a), get<2>(a), get<0>(a), get<2>(a), get<3>(a));
  auto tensor_b = create_desc_shape_range(get<0>(b), get<1>(b), get<2>(b), get<0>(b), get<2>(b), get<3>(b));
  BatchMatMul op;
  op.UpdateInputDesc("x1", tensor_a);
  op.UpdateInputDesc("x2", tensor_b);
  op.SetAttr("adj_x1", trans_a);
  op.SetAttr("adj_x2", trans_b);
  return op;
}

void Operate(BatchMatMul &op) {
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

void Check(BatchMatMul &op, vector<int64_t> expected_shape, vector<pair<int64_t,int64_t>> expected_range) {
  auto output_desc = op.GetOutputDesc("y");

  auto shape = output_desc.GetShape().GetDims();
  vector<pair<int64_t,int64_t>> range;
  EXPECT_EQ(output_desc.GetShapeRange(range), GRAPH_SUCCESS);

  EXPECT_EQ(shape, expected_shape);
  EXPECT_EQ(range, expected_range);
}

TEST(BatchMatMulInferTest, StaticNormal1) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {3, 2, 5}, {{3, 3}, {2, 2}, {5, 5}});
}

TEST(BatchMatMulInferTest, StaticNormal2) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{2, 3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, 5}, {{2, 2}, {3, 3}, {2, 2}, {5, 5}});
}

TEST(BatchMatMulInferTest, StaticNormal3) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{2, 3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{1, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, 5}, {{2, 2}, {3, 3}, {2, 2}, {5, 5}});
}

TEST(BatchMatMulInferTest, StaticNormal4) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{2, 3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, 5}, {{2, 2}, {3, 3}, {2, 2}, {5, 5}});
}