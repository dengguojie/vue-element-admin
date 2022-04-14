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


using namespace std;
using namespace ge;
using namespace op;

#define OP_TUPLE tuple<vector<int64_t>, DataType, Format, vector<pair<int64_t,int64_t>>>
#define PASS true
#define FAILED false

class BatchMatMulInferSliceTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "BatchMatMulInferSliceTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BatchMatMulInferSliceTest TearDown" << std::endl;
  }
};


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

void Operate(BatchMatMul &op, bool expected_result = PASS) {
  auto verify_ret = op.VerifyAllAttr(true);
  auto infer_ret = op.InferShapeAndType();

  // check result
  auto ret = (verify_ret == GRAPH_FAILED || infer_ret == GRAPH_FAILED) ? GRAPH_FAILED : GRAPH_SUCCESS;
  if (expected_result == PASS){
    EXPECT_EQ(ret, GRAPH_SUCCESS);
  } else {
    EXPECT_EQ(ret, GRAPH_FAILED);
  }
}

void Check(BatchMatMul &op, vector<int64_t> expected_shape, vector<pair<int64_t,int64_t>> expected_range) {
  auto output_desc = op.GetOutputDesc("y");

  auto shape = output_desc.GetShape().GetDims();
  vector<pair<int64_t,int64_t>> range;
  EXPECT_EQ(output_desc.GetShapeRange(range), GRAPH_SUCCESS);

  EXPECT_EQ(shape, expected_shape);
  EXPECT_EQ(range, expected_range);
}

// b tensor has no batch
TEST(BatchMatMulInferTest, StaticNormal1) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {3, 2, 5}, {});
}

// num_dima<num_dimb, batch_a=batch_b
TEST(BatchMatMulInferTest, StaticNormal2) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{2, 3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, 5}, {});
}

// num_dima>num_dimb, batch_b=1
TEST(BatchMatMulInferTest, StaticNormal3) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{2, 3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{1, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, 5}, {});
}

// num_dima>num_dimb, batch_a=batch_b
TEST(BatchMatMulInferTest, StaticNormal4) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{2, 3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, 5}, {});
}

// num_dima=num_dimb, batch_a=1
TEST(BatchMatMulInferTest, StaticNormal5) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{1, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {3, 2, 5}, {});
}

// num_dima = 1
TEST(BatchMatMulInferTest, StaticNormal6) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {3, 5}, {});
}

// num_dimb = 1
TEST(BatchMatMulInferTest, StaticNormal7) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{1, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{4}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {1, 2}, {});
}

// num_dima<num_dimb, batch_a=batch_b
TEST(BatchMatMulInferTest, DynamicNormal1) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{2, 3, 4, -1}, DT_FLOAT16, FORMAT_ND, {{2, 2}, {3, 3}, {4, 4}, {3, 6}}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, -1}, {{2, 2}, {3, 3}, {2, 2}, {3, 6}});
}

// num_dima>num_dimb, batch_b=1
TEST(BatchMatMulInferTest, DynamicNormal2) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{2, 3, 2, -1}, DT_FLOAT16, FORMAT_ND, {{2, 2}, {3, 3}, {2, 2}, {1, 3}}},
                                OP_TUPLE{{1, -1, 5}, DT_FLOAT16, FORMAT_ND, {{1, 1}, {2, 4}, {5, 5}}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, 2, 5}, {});
}

// num_dima>num_dimb, batch_a=batch_b
TEST(BatchMatMulInferTest, DynamicNormal3) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{2, 3, -1, 4}, DT_FLOAT16, FORMAT_ND, {{2, 2}, {3, 3}, {1, 3}, {4, 4}}},
                                OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {2, 3, -1, 5}, {{2, 2}, {3, 3}, {1, 3}, {5, 5}});

}

// num_dima=num_dimb, batch_a=-1
TEST(BatchMatMulInferTest, DynamicNormal4) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{-1, 2, 4}, DT_FLOAT16, FORMAT_ND, {{1, 3}, {2, 2}, {4, 4}}},
                                OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {3, 2, 5}, {});
}

// num_dima=num_dimb, batch_b=-1
TEST(BatchMatMulInferTest, DynamicNormal5) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{-1, 4, 5}, DT_FLOAT16, FORMAT_ND, {{1, 3}, {4, 4}, {5, 5}}},                                
                                false, false);

  Operate(op);

  Check(op, {3, 2, 5}, {});
}

// num_dima=num_dimb, batch_b=-1, batch_a=-1, rang_a.first=1, range_b.first=1
TEST(BatchMatMulInferTest, DynamicNormal6) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{-1, 2, 4}, DT_FLOAT16, FORMAT_ND, {{1, 6}, {2, 2}, {4, 4}}},
                                OP_TUPLE{{-1, 4, 5}, DT_FLOAT16, FORMAT_ND, {{1, 3}, {4, 4}, {5, 5}}},                                
                                false, false);

  Operate(op);

  Check(op, {-1, 2, 5}, {{1, 6}, {2, 2}, {5, 5}});
}

// num_dima=num_dimb, batch_b=-1, batch_a=-1, rang_a.first>1, range_b.first>1
TEST(BatchMatMulInferTest, DynamicNormal7) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{-1, 2, 4}, DT_FLOAT16, FORMAT_ND, {{2, 6}, {2, 2}, {4, 4}}},
                                OP_TUPLE{{-1, 4, 5}, DT_FLOAT16, FORMAT_ND, {{3, 5}, {4, 4}, {5, 5}}},                                
                                false, false);

  Operate(op);

  Check(op, {-1, 2, 5}, {{3, 5}, {2, 2}, {5, 5}});
}

// num_dima=num_dimb, -2
TEST(BatchMatMulInferTest, DynamicNormal8) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{-2}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {-2}, {});
}

// num_dima = 1
TEST(BatchMatMulInferTest, DynamicNormal9) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{-1, 4, 5}, DT_FLOAT16, FORMAT_ND, {{3, 5}, {4, 4}, {5, 5}}},
                                false, false);

  Operate(op);

  Check(op, {-1,5}, {{3, 5}, {5, 5}});
}

// num_dima = 1
TEST(BatchMatMulInferTest, DynamicNorma20) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{3, 4, -1}, DT_FLOAT16, FORMAT_ND, {{3, 3}, {4, 4}, {2, 5}}},
                                false, false);

  Operate(op);

  Check(op, {3,-1}, {{3, 3}, {2, 5}});
}

// num_dimb = 1
TEST(BatchMatMulInferTest, DynamicNorma21) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{-1, 4, 5}, DT_FLOAT16, FORMAT_ND, {{3, 5}, {4, 4}, {5, 5}}},
                                OP_TUPLE{{5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op);

  Check(op, {-1,4}, {{3, 5}, {4, 4}});
}

// k_a!=k_b
TEST(BatchMatMulInferTest, supportcheckerror1) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{3, 6, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op, FAILED);
}

// k_b=-1, k_a not in range_b
TEST(BatchMatMulInferTest, supportcheckerror2) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{3, -1, 5}, DT_FLOAT16, FORMAT_ND, {{6, 7}}},
                                false, false);

  Operate(op, FAILED);
}

// k_a=-1, k_b not in range_a
TEST(BatchMatMulInferTest, supportcheckerror3) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, -1, 5}, DT_FLOAT16, FORMAT_ND, {{6, 7}}},
                                OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op, FAILED);
}

// k_b=-1 k_a=-1, range_a and range_b has no intersection
TEST(BatchMatMulInferTest, supportcheckerror4) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 4, -1}, DT_FLOAT16, FORMAT_ND, {{6, 7}}},
                                OP_TUPLE{{3, -1, 5}, DT_FLOAT16, FORMAT_ND, {{3, 4}}},
                                false, false);

  Operate(op, FAILED);
}

// num_dima=num_dimb, batch_b=-1, batch_a=-1, rang_a.first>1, range_b.first>1
TEST(BatchMatMulInferTest, supportcheckerror5) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{-1, 2, 4}, DT_FLOAT16, FORMAT_ND, {{4, 6}}},
                                OP_TUPLE{{-1, 4, 5}, DT_FLOAT16, FORMAT_ND, {{2, 3}}},                                
                                false, false);

  Operate(op, FAILED);
}

// k_b=-1, batch_a>1, batch_b>1, batch_a!=batch_b
TEST(BatchMatMulInferTest, supportcheckerror6) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{4, -1, 5}, DT_FLOAT16, FORMAT_ND, {{2, 7}}},
                                false, false);

  Operate(op, FAILED);
}

// num_dima=num_dimb, batch_a=-1, batch_b not in range_a
TEST(BatchMatMulInferTest, supportcheckerror7) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{-1, 2, 4}, DT_FLOAT16, FORMAT_ND, {{1, 3}}},
                                OP_TUPLE{{5, 4, 5}, DT_FLOAT16, FORMAT_ND, {}},
                                false, false);

  Operate(op, FAILED);
}

// num_dima=num_dimb, batch_b=-1, batch_a not in range_b
TEST(BatchMatMulInferTest, supportcheckerror8) {
  auto op = CreateBatchMatMulOp(OP_TUPLE{{3, 2, 4}, DT_FLOAT16, FORMAT_ND, {}},
                                OP_TUPLE{{-1, 4, 5}, DT_FLOAT16, FORMAT_ND, {{4, 6}}},                                
                                false, false);

  Operate(op, FAILED);
}

// cut batch in NZ
TEST_F(BatchMatMulInferSliceTest, split_test0) {
  ge::op::BatchMatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("adj_x1", false);
  op.SetAttr("adj_x2", false);

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
TEST_F(BatchMatMulInferSliceTest, split_test1) {
  ge::op::BatchMatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({16, 4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("adj_x1", false);
  op.SetAttr("adj_x2", false);

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
TEST_F(BatchMatMulInferSliceTest, split_test2) {
  ge::op::BatchMatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({8, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{8, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({8, 4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{8, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({8, 4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{8, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("adj_x1", false);
  op.SetAttr("adj_x2", false);

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
TEST_F(BatchMatMulInferSliceTest, split_test3) {
  ge::op::BatchMatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("adj_x1", false);
  op.SetAttr("adj_x2", false);

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
TEST_F(BatchMatMulInferSliceTest, split_test4) {
  ge::op::BatchMatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({4, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{4, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({4, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{4, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("adj_x1", false);
  op.SetAttr("adj_x2", false);

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
TEST_F(BatchMatMulInferSliceTest, split_test5) {
  ge::op::BatchMatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({16, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({16, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 64, 64}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y", create_desc_with_ori({16, 32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{16, 32, 64}, ge::FORMAT_ND));
  op.SetAttr("adj_x1", false);
  op.SetAttr("adj_x2", false);

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
