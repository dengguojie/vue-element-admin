#include <tuple>
#include <cstdlib>

#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "op_proto_test_util.h"

using namespace std;
using namespace ge;
using namespace op;

#define OP_TUPLE tuple<vector<int64_t>, DataType, Format, vector<pair<int64_t,int64_t>>>
#define RES_TUPLE tuple<vector<int64_t>, vector<pair<int64_t,int64_t>>>
#define CASE_TUPLE tuple<OP_TUPLE, OP_TUPLE, OP_TUPLE, bool, bool, RES_TUPLE>

vector<CASE_TUPLE> testcase_batchmatmulv2 = {
    /* no bias */
    CASE_TUPLE{OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {{3, 3}, {2, 2}, {5, 5}}}},
    CASE_TUPLE{OP_TUPLE{{2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {{3, 3}, {2, 2}, {5, 5}}}},
    CASE_TUPLE{OP_TUPLE{{2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {{3, 3}, {2, 2}, {5, 5}}}},
    CASE_TUPLE{OP_TUPLE{{1, 2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {{3, 3}, {2, 2}, {5, 5}}}},
    CASE_TUPLE{OP_TUPLE{{4, 3, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-2}, {}}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-2}, {}}},
    // broadcast
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{4, 9}, {1, -1}, {1, -1}}},
               OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 5}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{4, 5}, {1, -1}, {7, 7}}}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 8}, {1, -1}, {1, -1}}},
               OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 6}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{1, 8}, {1, -1}, {7, 7}}}},
    CASE_TUPLE{OP_TUPLE{{1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 1}, {1, -1}, {1, -1}}},
               OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 5}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{2, 5}, {1, -1}, {7, 7}}}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 5}, {1, -1}, {1, -1}}},
               OP_TUPLE{{1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 1}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{2, 5}, {1, -1}, {7, 7}}}},
    /* with bias */
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{4, 3, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{4, 9}, {7, 7}, {1, -1}}},
               OP_TUPLE{{-1, -1}, DT_FLOAT16, FORMAT_ND, {{1, -1}, {7, 9}}}, OP_TUPLE{{8}, DT_FLOAT16, FORMAT_ND, {}},
               false, false, RES_TUPLE{{-1, 7, 8}, {{4, 9}, {7, 7}, {8, 8}}}},
    /* dts */
    CASE_TUPLE{OP_TUPLE{{-1, 3, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 5}, {3, 3}, {1, 5}, {1, 5}}},
               OP_TUPLE{{-1, -1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 80}, {2, 80}, {2, 80}, {2, 80}}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false,
               RES_TUPLE{{-1, 3, -1, 5}, {{2, 80}, {3, 3}, {1, 5}, {5, 5}}}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               true,
               false,
               RES_TUPLE{{-2}, {}}},
    CASE_TUPLE{OP_TUPLE{{5, 20}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, 60}, DT_FLOAT16, FORMAT_ND, {{1, 4}, {1, 1}, {16, 16}, {16, 16}}},
               {},
               true,
               false,
               RES_TUPLE{{20, 60}, {{20, 20}, {60, 60}}}},
};

BatchMatMulV2 CreateBatchMatMulV2Op(OP_TUPLE a, OP_TUPLE b, OP_TUPLE bias,
                                    bool trans_a, bool trans_b) {
  auto tensor_a = create_desc_shape_range(get<0>(a), get<1>(a), get<2>(a), get<0>(a), get<2>(a), get<3>(a));
  auto tensor_b = create_desc_shape_range(get<0>(b), get<1>(b), get<2>(b), get<0>(b), get<2>(b), get<3>(b));
  TensorDesc tensor_bias;
  auto shape_bias = get<0>(bias);
  if (!shape_bias.empty()) {
    tensor_bias = create_desc_shape_range(get<0>(bias), get<1>(bias), get<2>(bias), get<0>(bias), get<2>(bias), get<3>(bias));
  }

  BatchMatMulV2 op;
  op.UpdateInputDesc("x1", tensor_a);
  op.UpdateInputDesc("x2", tensor_b);
  if (!shape_bias.empty()) {
    op.UpdateInputDesc("bias", tensor_bias);
  }
  op.SetAttr("adj_x1", trans_a);
  op.SetAttr("adj_x2", trans_b);
  return op;
}

void Operate(BatchMatMulV2 &op) {
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

void Check(BatchMatMulV2 &op, const RES_TUPLE &expected) {
  auto output_desc = op.GetOutputDesc("y");

  auto shape = output_desc.GetShape().GetDims();
  vector<pair<int64_t,int64_t>> range;
  EXPECT_EQ(output_desc.GetShapeRange(range), GRAPH_SUCCESS);

  EXPECT_EQ(shape, get<0>(expected));
  EXPECT_EQ(range, get<1>(expected));
}

class BatchMatMulV2Test :
  public testing::TestWithParam<CASE_TUPLE> {
public:
  BatchMatMulV2Test() {}
  static void SetUpTestSuite() {
    setenv("GLOBAL_LOG_LEVEL", "0", true);
  }
  static void TearDownTestSuite() {
    unsetenv("GLOBAL_LOG_LEVEL");
  }
};

TEST_P(BatchMatMulV2Test, General) {
  auto op = CreateBatchMatMulV2Op(std::get<0>(GetParam()),
                             std::get<1>(GetParam()),
                             std::get<2>(GetParam()),
                             std::get<3>(GetParam()),
                             std::get<4>(GetParam()));

  Operate(op);

  Check(op, get<5>(GetParam()));
}

INSTANTIATE_TEST_CASE_P(DynamicShape,
                        BatchMatMulV2Test,
                        testing::ValuesIn(testcase_batchmatmulv2));
