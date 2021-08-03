#include <tuple>
#include <cstdlib>

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

using namespace std;
using namespace ge;
using namespace op;

#define OP_TUPLE tuple<vector<int64_t>, DataType, Format, vector<pair<int64_t,int64_t>>>
#define RES_TUPLE tuple<vector<int64_t>, vector<pair<int64_t,int64_t>>, bool>
#define CASE_TUPLE tuple<OP_TUPLE, OP_TUPLE, OP_TUPLE, bool, bool, RES_TUPLE>
#define PASS true
#define FAILED false

vector<CASE_TUPLE> testcase_batchmatmulv2 = {
    /* no bias */
    CASE_TUPLE{OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{1, 2, 4}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{3, 2, 5}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{4, 3, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-2}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{-2}, {}, PASS}},
    // broadcast
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{4, 9}, {1, -1}, {1, -1}}},
               OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 5}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{4, 5}, {1, -1}, {7, 7}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 8}, {1, -1}, {1, -1}}},
               OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 6}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{1, 8}, {1, -1}, {7, 7}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 1}, {1, -1}, {1, -1}}},
               OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 5}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{2, 5}, {1, -1}, {7, 7}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 5}, {1, -1}, {1, -1}}},
               OP_TUPLE{{1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 1}, {1, -1}, {7, 7}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, 7}, {{2, 5}, {1, -1}, {7, 7}}, PASS}},
    /* with bias */
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{4, 3, 5}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}}, false, false, RES_TUPLE{{-2}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{4, 9}, {7, 7}, {1, -1}}},
               OP_TUPLE{{-1, -1}, DT_FLOAT16, FORMAT_ND, {{1, -1}, {7, 9}}}, OP_TUPLE{{8}, DT_FLOAT16, FORMAT_ND, {}},
               false, false, RES_TUPLE{{-1, 7, 8}, {{4, 9}, {7, 7}, {8, 8}}, PASS}},
    /* dts */
    CASE_TUPLE{OP_TUPLE{{-1, 3, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1, 5}, {3, 3}, {1, 5}, {1, 5}}},
               OP_TUPLE{{-1, -1, -1, -1}, DT_FLOAT16, FORMAT_ND, {{2, 80}, {2, 80}, {2, 80}, {2, 80}}},
               OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}}, false, false,
               RES_TUPLE{{-1, 3, -1, 5}, {{2, 80}, {3, 3}, {1, 5}, {5, 5}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, -1}, DT_FLOAT16, FORMAT_ND, {{5, 5}, {1, 60}}},
               {},
               true,
               false,
               RES_TUPLE{{-2}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{5, 20}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{5, 60}, DT_FLOAT16, FORMAT_ND, {{1, 4}, {1, 1}, {16, 16}, {16, 16}}},
               {},
               true,
               false,
               RES_TUPLE{{20, 60}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-1, 13, 16}, DT_FLOAT16, FORMAT_ND, {{8, -1}, {13, 13}, {16, 16}}},
               OP_TUPLE{{-1, 13, 16}, DT_FLOAT16, FORMAT_ND, {{8, -1}, {13, 13}, {16, 16}}},
               {},
               true,
               false,
               RES_TUPLE{{-1, 16, 16}, {{8, -1}, {16, 16}, {16, 16}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{1024, 8, -1, -1}, DT_FLOAT16, FORMAT_ND, {{1024, 1024}, {8, 8}, {1, 536870912}, {1, -1}}},
               OP_TUPLE{{1024, 8, -1, 64}, DT_FLOAT16, FORMAT_ND, {{1, 536870912}, {1, 536870912}, {2, -1}, {1, 536870912}}},
               {},
               false,
               false,
               RES_TUPLE{{1024, 8, -1, 64}, {{1024, 1024}, {8, 8}, {1, 536870912}, {64, 64}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-1, 13, 16}, DT_FLOAT16, FORMAT_ND, {{-1, -1}, {13, 13}, {16, 16}}},
               OP_TUPLE{{-1, 13, 16}, DT_FLOAT16, FORMAT_ND, {{-1, -1}, {13, 13}, {16, 16}}},
               {},
               true,
               false,
               RES_TUPLE{{}, {}, FAILED}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{2048, 64, 300}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               true,
               RES_TUPLE{{2048, -1, 64}, {{2048, 2048}, {1, -1}, {64, 64}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-1, -1, -1, 512}, DT_FLOAT16, FORMAT_ND, {{1, -1}, {2, 3}, {1, 2}, {512, 512}}},
               OP_TUPLE{{512, -1}, DT_FLOAT16, FORMAT_ND, {{512, 52}, {3, 4}}},
               {},
               false,
               false,
               RES_TUPLE{{-1, -1, -1, -1}, {{1, -1}, {2, 3}, {1, 2}, {3, 4}}, PASS}},
    CASE_TUPLE{OP_TUPLE{{0, 10}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{10, 20}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{0, 20}, {}, PASS}},
    CASE_TUPLE{OP_TUPLE{{-3, 10}, DT_FLOAT16, FORMAT_ND, {}},
               OP_TUPLE{{10, 20}, DT_FLOAT16, FORMAT_ND, {}},
               {},
               false,
               false,
               RES_TUPLE{{}, {}, FAILED}},
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

void Operate(BatchMatMulV2 &op, bool expected_result) {
  auto ret = op.InferShapeAndType();

  if (expected_result == PASS) {
    EXPECT_EQ(ret, GRAPH_SUCCESS);
  } else {
    EXPECT_EQ(ret, GRAPH_FAILED);
  }
}

void Check(BatchMatMulV2 &op, const RES_TUPLE &expected) {
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

  auto result_tuple = get<5>(GetParam());
  auto expected_result = get<2>(result_tuple);

  Operate(op, expected_result);

  if (expected_result == PASS) {
    Check(op, result_tuple);
  }
}

INSTANTIATE_TEST_CASE_P(DynamicShape,
                        BatchMatMulV2Test,
                        testing::ValuesIn(testcase_batchmatmulv2));


// cut batch in NZ
TEST_F(BatchMatMulV2Test, split_test0) {
  ge::op::BatchMatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{2, 8}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{2, 8}, {}, {}, {}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);
}

// cut n in NZ
TEST_F(BatchMatMulV2Test, split_test1) {
  ge::op::BatchMatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({16, 4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 1}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);

  std::vector<std::vector<int64_t>> expect_x2_data_slice = {{}, {0, 1}, {}, {}, {}};
  EXPECT_EQ(expect_x2_data_slice, x2_data_slice);
}

// cut m in NZ
TEST_F(BatchMatMulV2Test, split_test2) {
  ge::op::BatchMatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({8, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{8, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({8, 4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{8, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({8, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{8, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 1}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{}, {}, {0, 1}, {}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);
}

// cut batch in ND
TEST_F(BatchMatMulV2Test, split_test3) {
  ge::op::BatchMatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice = {{4, 8}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{4, 8}, {}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);

  ge::GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);

  std::vector<std::vector<int64_t>> expect_x2_data_slice = {{4, 8}, {}, {}};
  EXPECT_EQ(expect_x2_data_slice, x2_data_slice);
}

// cut n in ND
TEST_F(BatchMatMulV2Test, split_test4) {
  ge::op::BatchMatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({4, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{4, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({4, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{4, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 15}};
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
TEST_F(BatchMatMulV2Test, split_test5) {
  ge::op::BatchMatMulV2 op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {16, 31}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{}, {16, 31}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);
}

