#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"
#include "plugin_test_utils.h"

class gather_point_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_point_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_point_test TearDown" << std::endl;
  }
};

TEST_F(gather_point_test, gather_point_test_1) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/gatherpoint_case_1.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}