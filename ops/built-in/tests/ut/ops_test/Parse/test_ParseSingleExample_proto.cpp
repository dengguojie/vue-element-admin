#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "parsing_ops.h"
using namespace ge;
using namespace op;

class parsesingleexample : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "parse_single_example SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "parse_single_example TearDown" << std::endl;
  }
};

TEST_F(parsesingleexample, parsesingleexample_infer_shape_01) {
  ge::op::ParseSingleExample op;
  op.UpdateInputDesc("serialized", create_desc({}, ge::DT_STRING));
  op.SetAttr("num_sparse", 1);
  
  std::vector<std::string> sparse_keys = {"sparse_keys"};
  op.SetAttr("sparse_keys", sparse_keys);

  std::vector<std::string> dense_keys = {"dense_keys"};
  op.SetAttr("dense_keys", dense_keys);
  
  std::vector<ge::DataType> sparse_types={DT_FLOAT};
  op.SetAttr("sparse_types", sparse_types);
  
  std::vector<ge::DataType> dense_types={DT_FLOAT};
  op.SetAttr("Tdense", dense_types);

  std::vector<std::vector<int64_t>> dense_shapes = {{-1}};
  op.SetAttr("dense_shapes", dense_shapes);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(parsesingleexample, parsesingleexample_infer_shape_02) {
  ge::op::ParseSingleExample op;
  op.UpdateInputDesc("serialized", create_desc({1}, ge::DT_STRING));
  op.SetAttr("num_sparse", 1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parsesingleexample, parsesingleexample_infer_shape_03) {
 ge::op::ParseSingleExample op;
  op.UpdateInputDesc("serialized", create_desc({}, ge::DT_STRING));
  op.SetAttr("num_sparse", 2);
  
  std::vector<std::string> sparse_keys = {"sparse_keys"};
  op.SetAttr("sparse_keys", sparse_keys);

  std::vector<std::string> dense_keys = {"dense_keys"};
  op.SetAttr("dense_keys", dense_keys);
  
  std::vector<ge::DataType> sparse_types={DT_FLOAT};
  op.SetAttr("sparse_types", sparse_types);
  
  std::vector<ge::DataType> dense_types={DT_FLOAT};
  op.SetAttr("Tdense", dense_types);

  std::vector<std::vector<int64_t>> dense_shapes = {{-1}};
  op.SetAttr("dense_shapes", dense_shapes);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  
}


