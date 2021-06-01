#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
using namespace std;
using namespace aicpu;

class TEST_SPARSETODENSE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, Validate)            \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  NodeDefBuilder(node_def.get(), "SparseToDense", "SparseToDense")     \
      .Input({"indices", data_types[0], shapes[0], datas[0]})          \
      .Input({"out_put_shape", data_types[1], shapes[1], datas[1]})    \
      .Input({"values", data_types[2], shapes[2], datas[2]})         \
	  .Input({"Default_values", data_types[3], shapes[3], datas[3]})  \
      .Output({"y", data_types[4], shapes[4], datas[4]})              \
      .Attr("validate_indices", Validate);

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_SPARSETODENSE_UT, TestSparseToDense_##aicpu_type) {            \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type, aicpu_type, aicpu_type};  \
    vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};          \
    base_type input1[3] = {0,1,1};                                               \
    base_type input2[3] = {2,2,2};                                               \
    base_type input3[1] = {1};                                               \
    base_type input4[1] = {5};                                               \
    base_type output[8] = {(base_type)0};                                     \
    int32_t concat_dim = 0;                                                    \
    bool Validate= true;                                                      \
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output}; \
    CREATE_NODEDEF(shapes, data_types, datas, Validate);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
  }

TEST_F(TEST_SPARSETODENSE_UT, Host2) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3,1}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[3] = {2,2,2};
  int64_t input3[1] = {1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host3) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {4}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2,1};
  int64_t input3[1] = {1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host4) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host5) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[1] = {0};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host6) {
  vector<DataType> data_types = {DT_UINT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[3] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host7) {
  vector<DataType> data_types = {DT_UINT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}, {2,2,2}};
  int64_t input1;
  int64_t input2;
  int64_t input3;
  int64_t input4;
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host8) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host9) {
  vector<DataType> data_types = {DT_INT64, DT_UINT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host10) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3, 4}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host11) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{4, 1}, {1}, {4}, {}, {2,2,2}};
  int64_t input1[3] = {0,0,0};
  int64_t input2[4] = {0,0,0};
  int64_t input3[2] = {0,0};
  int64_t input4[1] = {0};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host12) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host13) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {4,5,8}, {2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}


TEST_F(TEST_SPARSETODENSE_UT, Host14) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= false;
  vector<void *> datas = {(void *)nullptr, (void *)input2, (void *)&input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host15) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1, 3}, {3}, {1}, {}, {2,2,2}};
  int64_t input1[3] = {0,1,1};
  int64_t input2[4] = {2,2,2};
  int64_t input3[2] = {1,1};
  int64_t input4[1] = {5};
  int64_t output[8] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)nullptr, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
//ADD_CASE(int64_t, DT_INT64)
//ADD_CASE(int32_t, DT_INT32)

