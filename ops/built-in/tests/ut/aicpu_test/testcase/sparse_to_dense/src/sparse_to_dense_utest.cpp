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

TEST_F(TEST_SPARSETODENSE_UT, Host16) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int64_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int64_t input2[2] = {10,2};
  int64_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  int64_t input4[1] = {9};
  int64_t output[10][2] = {(int64_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host17) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int32_t input2[2] = {10,2};
  int32_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  int32_t input4[1] = {9};
  int32_t output[10][2] = {(int32_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host18) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int32_t input2[2] = {10,2};
  int8_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  int8_t input4[1] = {9};
  int8_t output[10][2] = {(int8_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host19) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int32_t input2[2] = {10,2};
  uint8_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  uint8_t input4[1] = {9};
  uint8_t output[10][2] = {(uint8_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host20) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int32_t input2[2] = {10,2};
  int16_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  int16_t input4[1] = {9};
  int16_t output[10][2] = {(int16_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host21) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int32_t input2[2] = {10,2};
  uint16_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  uint16_t input4[1] = {9};
  uint16_t output[10][2] = {(uint16_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host22) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int32_t input2[2] = {10,2};
  Eigen::half input3[10] = {(Eigen::half)1.0,(Eigen::half)2.0,(Eigen::half)3.0,(Eigen::half)4.0,(Eigen::half)5.0,
                            (Eigen::half)6.0,(Eigen::half)7.0,(Eigen::half)8.0,(Eigen::half)9.0,(Eigen::half)10.0};
  Eigen::half input4[1] = {(Eigen::half)9.0};
  Eigen::half output[10][2] = {(Eigen::half)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host23) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{1,0},{2,0},{3,0},{4,0},{5,0},{5,1},{6,1},{7,1},{8,1},{9,1}};
  int32_t input2[2] = {10,2};
  float input3[10] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};
  float input4[1] = {9.0};
  float output[10][2] = {(float)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSETODENSE_UT, Host24) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{6,1},{7,1},{8,1},{9,1},{10,1},{1,0},{2,0},{3,0},{4,0},{5,0}};
  int32_t input2[2] = {10,2};
  int32_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  int32_t input4[1] = {9};
  int32_t output[10][2] = {(int32_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host25) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{6,1},{6,1},{6,1},{9,1},{10,1},{1,0},{2,0},{3,0},{4,0},{5,0}};
  int32_t input2[2] = {10,2};
  int32_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  int32_t input4[1] = {9};
  int32_t output[10][2] = {(int32_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSETODENSE_UT, Host26) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10,2}, {2}, {10}, {}, {10,2}};
  int32_t input1[10][2] = {{6,2},{6,1},{6,1},{9,1},{10,1},{1,0},{2,0},{3,0},{4,0},{5,0}};
  int32_t input2[2] = {10,2};
  int32_t input3[10] = {1,2,3,4,5,6,7,8,9,10};
  int32_t input4[1] = {9};
  int32_t output[10][2] = {(int32_t)0};
  bool Validate= true;
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)&input4 ,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, Validate);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

ADD_CASE(int64_t, DT_INT64)
ADD_CASE(int32_t, DT_INT32)

