#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "vector_search.h"

class ProdVirialSeAProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdVirialSeA Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdVirialSeA Proto Test TearDown" << std::endl;
  }
};

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAVerifyTest_1) {
  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc({1, 6782976}, ge::DT_FLOAT));
  op.UpdateInputDesc("in_deriv", create_desc({1, 20348928}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij", create_desc({1, 5087232}, ge::DT_FLOAT));
  op.UpdateInputDesc("nlist", create_desc({1, 1695744}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({3}, ge::DT_INT32));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  op.SetAttr("nall", 28328);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAVerifyTest_2) {
  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc({1, 6782976}, ge::DT_FLOAT));
  op.UpdateInputDesc("in_deriv", create_desc({1, 20348928}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij", create_desc({2, 5087232}, ge::DT_FLOAT));
  op.UpdateInputDesc("nlist", create_desc({1, 1695744}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({3}, ge::DT_INT32));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  op.SetAttr("nall", 28328);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAVerifyTest_3) {
  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc({1, 6782976}, ge::DT_FLOAT));
  op.UpdateInputDesc("in_deriv", create_desc({1, 20348928}, ge::DT_FLOAT));
  op.UpdateInputDesc("rij", create_desc({1, 5087232}, ge::DT_FLOAT16));
  op.UpdateInputDesc("nlist", create_desc({1, 1695744}, ge::DT_INT32));
  op.UpdateInputDesc("natoms", create_desc({3}, ge::DT_INT32));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  op.SetAttr("nall", 28328);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ProdVirialSeAProtoTest, ProdVirialSeAInferShapeTest_1) {
  ge::op::ProdVirialSeA op;
  op.UpdateInputDesc("net_deriv", create_desc_shape_range({1, 6782976}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 6782976},
                                                          ge::FORMAT_ND, {{1, 1}, {6782976, 6782976}}));
  op.UpdateInputDesc("in_deriv", create_desc_shape_range({1, 20348928}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 20348928},
                                                         ge::FORMAT_ND, {{1, 1}, {20348928, 20348928}}));
  op.UpdateInputDesc("rij", create_desc_shape_range({1, 5087232}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 5087232},
                                                    ge::FORMAT_ND, {{1, 1}, {5087232, 5087232}}));
  op.UpdateInputDesc("nlist", create_desc_shape_range({1, 1695744}, ge::DT_INT32, ge::FORMAT_ND, {1, 1695744},
                                                      ge::FORMAT_ND, {{1, 1}, {1695744, 1695744}}));
  op.UpdateInputDesc("natoms", create_desc_shape_range({3}, ge::DT_INT32, ge::FORMAT_ND, {3}, ge::FORMAT_ND,
                                                       {{3, 3}}));
  op.SetAttr("n_a_sel", 138);
  op.SetAttr("n_r_sel", 0);
  op.SetAttr("nall", 28328);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  {
    auto output_desc = op.GetOutputDescByName("virial");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {1, 9};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 1}, {9, 9}};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);
    EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }

  {
    auto output_desc = op.GetOutputDescByName("atom_virial");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {1, 254952};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 1}, {254952, 254952}};
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);
    EXPECT_EQ(output_shape_range, expected_output_shape_range);
  }
}