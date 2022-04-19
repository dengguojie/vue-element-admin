#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tensorflow_parser.h"
#include "../onnx/parser_common.h"

using namespace ge;

class batchmatmulv3_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "batchmatmulv3_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batchmatmulv3_plugin_test TearDown" << std::endl;
  }
};

TEST_F(batchmatmulv3_plugin_test, batchmatmulv3_plugin_test_1) {
  CleanGlobal();
  ge::Graph graph;

  std::string case_dir = __FILE__;
  std::size_t idx = case_dir.find_last_of("/");
  case_dir = case_dir.substr(0, idx);
  std::string model_file = case_dir + "/batchmatmulv3_case_1.pb";

  auto status = aclgrphParseTensorFlow(model_file.c_str(), graph);
  CleanGlobal();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(batchmatmulv3_plugin_test, batchmatmulv2_plugin_test_1) {
  CleanGlobal();
  ge::Graph graph;

  std::string case_dir = __FILE__;
  std::size_t idx = case_dir.find_last_of("/");
  case_dir = case_dir.substr(0, idx);
  std::string model_file = case_dir + "/batchmatmulv2_case_1.pb";

  auto status = aclgrphParseTensorFlow(model_file.c_str(), graph);
  CleanGlobal();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

}