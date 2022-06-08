/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_dsl_transdata.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include <gtest/gtest.h>

#include "graph/utils/op_desc_utils.h"
#include "op_tiling/vector_tiling.h"
#include "op_tiling/transdata_dsl_general.h"
#include "op_tiling/transdata_dsl_borrow.h"
#include "op_tiling/tiling_handler.h"

using namespace std;
using namespace optiling;
using namespace optiling::transdata_dsl;

class TransdataTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "TransdataTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "TransdataTiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
  return result;
}

static void contruct_tensor(ge::OpDescPtr& op_desc, const std::vector<int64_t>& shape, const ge::DataType dtype,
                            bool is_input=true, ge::Format format=ge::FORMAT_ND) {
  ge::GeTensorDesc tensor;
  tensor.SetShape(ge::GeShape(shape));
  tensor.SetFormat(format);
  tensor.SetDataType(dtype);
  if (is_input) {
    op_desc->AddInputDesc(tensor);
  } else {
    op_desc->AddOutputDesc(tensor);
  }
}

// Test Case: NCHW -> NC1HWC0, FP16, CONST, Compile
TEST_F(TransdataTiling, Forward_NCHW_5HD_Fp16_Const_Compile_General) {
//  string compileInfo = R"(
//    { "_src_pad_mode": [0, 2, 0],
//      "_src_pad_var": [1, 16, 1],
//      "_permute": [0, 1, 3, 2],
//      "_src_fuse": [0, 1, 2],
//      "_pattern": "Transdata",
//      "_common_info": [1, 16, 32, 1, 1],
//      "_ub_info": [[65280, 32640], [-1], [-1]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 1;
  op_compile_info.is_const_compile = 1;

  op_compile_info.src_pad_mode = {0, 2, 0};
  op_compile_info.src_pad_var = {1, 16, 1};
  op_compile_info.permute = {0, 1, 3, 2};
  op_compile_info.src_fuse = {0, 1, 2};
  op_compile_info.ub_info = {{65280, 32640}, {-1}, {-1}};

  vector<vector<int64_t>> inputs {
          {16, 192, 56*56}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56*56, 16}
  };

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataGeneral transdata(op_type, op_compile_info, runInfo, input, output, reshape, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NCHW -> NC1HWC0, FP32, CONST, Compile
TEST_F(TransdataTiling, Forward_NCHW_5HD_Fp32_Const_Compile_General) {
//  string compileInfo = R"(
//      { "_src_pad_mode": [0, 2, 0],
//        "_src_pad_var": [1, 16, 1],
//        "_permute": [0, 1, 3, 2],
//        "_src_fuse": [0, 1, 2],
//        "_pattern": "Transdata",
//        "_common_info": [1, 8, 32, 1, 1],
//        "_ub_info": [[16256, 16256], [-1], [-1]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 8;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 1;
  op_compile_info.is_const_compile = 1;

  op_compile_info.src_pad_mode = {0, 2, 0};
  op_compile_info.src_pad_var = {1, 16, 1};
  op_compile_info.permute = {0, 1, 3, 2};
  op_compile_info.src_fuse = {0, 1, 2};
  op_compile_info.ub_info = {{16256, 16256}, {-1}, {-1}};

  vector<vector<int64_t>> inputs {
          {16, 192, 56*56}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56*56, 16}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataGeneral transdata(op_type, op_compile_info, runInfo, input, output, reshape, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NHWC -> NC1HWC0, FP16, CONST, Compile
TEST_F(TransdataTiling, Forward_NHWC_5HD_Fp16_Const_Compile_General) {
//  string compileInfo = R"(
//    { "_src_pad_mode": [0, 0, 2],
//      "_src_pad_var": [1, 1, 16],
//      "_permute": [0, 2, 1, 3],
//      "_src_fuse": [0, 1, 3],
//      "_pattern": "Transdata",
//      "_common_info": [1, 16, 32, 1, 1],
//      "_ub_info": [[65280, 32640], [-1], [-1]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 1;
  op_compile_info.is_const_compile = 1;

  op_compile_info.src_pad_mode = {0, 0, 2};
  op_compile_info.src_pad_var = {1, 1, 16};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.ub_info = {{65280, 32640}, {-1}, {-1}};

  vector<vector<int64_t>> inputs {
          {16, 56*56, 192}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56*56, 16}
  };

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataGeneral transdata(op_type, op_compile_info, runInfo, input, output, reshape, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NHWC -> NC1HWC0, FP32, CONST, Compile
TEST_F(TransdataTiling, Forward_NHWC_5HD_Fp32_Const_Compile_General) {
//  string compileInfo = R"(
//      { "_src_pad_mode": [0, 0, 2],
//        "_src_pad_var": [1, 1, 16],
//        "_permute": [0, 2, 1, 3],
//        "_src_fuse": [0, 1, 3],
//        "_pattern": "Transdata",
//        "_common_info": [1, 8, 32, 1, 1],
//        "_ub_info": [[32640, 16256], [-1], [-1]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 8;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 1;
  op_compile_info.is_const_compile = 1;

  op_compile_info.src_pad_mode = {0, 0, 2};
  op_compile_info.src_pad_var = {1, 1, 16};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.ub_info = {{32640, 16256}, {-1}, {-1}};

  vector<vector<int64_t>> inputs {
          {16, 56*56, 192}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56*56, 16}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataGeneral transdata(op_type, op_compile_info, runInfo, input, output, reshape, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NHWC -> NC1HWC0, FP32, CONST, RUNTIME
TEST_F(TransdataTiling, Forward_NHWC_5HD_Fp32_Const_Runtime_General) {
//  string compileInfo = R"(
//        { "_src_pad_mode": [0, 0, 2],
//          "_src_pad_var": [1, 1, 16],
//          "_permute": [0, 2, 1, 3],
//          "_src_fuse": [0, 1, 3],
//          "_pattern": "Transdata",
//          "_common_info": [1, 8, 32, 1, 0],
//          "_ub_info": [[32640, 16256], [-1], [-1]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 8;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 1;
  op_compile_info.is_const_compile = 0;
  op_compile_info.const_block_dims.insert(pair<string, int>("123", 32));

  op_compile_info.src_pad_mode = {0, 0, 2};
  op_compile_info.src_pad_var = {1, 1, 16};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.ub_info = {{32640, 16256}, {-1}, {-1}};

  vector<vector<int64_t>> inputs {
          {16, 56*56, 192}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56*56, 16}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataGeneral transdata(op_type, op_compile_info, runInfo, input, output, reshape, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NHWC -> NC1HWC0, FP32, DYNAMIC
TEST_F(TransdataTiling, Forward_NHWC_5HD_Fp32_Dynamic_General) {
//  string compileInfo = R"(
//          { "_src_pad_mode": [0, 0, 2],
//            "_src_pad_var": [1, 1, 16],
//            "_permute": [0, 2, 1, 3],
//            "_src_fuse": [0, 1, 3],
//            "_pattern": "Transdata",
//            "_common_info": [1, 8, 32, 0, 0],
//            "_ub_info": [[32640, 16256], [65280], [65280]],
//            "_bh_x1x0": [2, 3],
//            "_bh_c1c0": [1, 4],
//            "_bh_permute": [0, 1, 5, 2, 3, 4],
//            "_bn_x1x0": [0, 1],
//            "_bn_c1c0": [2, 4],
//            "_bn_permute": [0, 5, 1, 2, 3, 4]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 8;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad_mode = {0, 0, 2};
  op_compile_info.src_pad_var = {1, 1, 16};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.ub_info = {{32640, 16256}, {65280}, {65280}};

  vector<vector<int64_t>> inputs {
          {16, 56, 56, 192}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56, 56, 16}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataGeneral transdata(op_type, op_compile_info, runInfo, input, output, reshape, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NCHW -> NC1HWC0, FP32, DYNAMIC
TEST_F(TransdataTiling, Forward_NCHW_5HD_Fp32_Dynamic_General) {
//  string compileInfo = R"(
//        { "_src_pad_mode": [0, 2, 0],
//          "_src_pad_var": [1, 16, 1],
//          "_permute": [0, 1, 3, 2],
//          "_src_fuse": [0, 1, 2],
//          "_pattern": "Transdata",
//          "_common_info": [1, 8, 32, 0, 0],
//          "_ub_info": [[16256, 16256], [-1], [-1]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 8;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad_mode = {0, 2, 0};
  op_compile_info.src_pad_var = {1, 16, 1};
  op_compile_info.permute = {0, 1, 3, 2};
  op_compile_info.src_fuse = {0, 1, 2};
  op_compile_info.ub_info = {{16256, 16256}, {-1}, {-1}};

  vector<vector<int64_t>> inputs {
          {16, 192, 56, 56}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56, 56, 16}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataGeneral transdata(op_type, op_compile_info, runInfo, input, output, reshape, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}


// Test Case: NC1HWC0 -> NHWC, Fp16, DYNAMIC
TEST_F(TransdataTiling, Backward_5HD_NHWC_Fp16_Dynamic_BN) {
//    string compileInfo = R"(
//          { "_src_pad_mode": [0, 0, 2],
//            "_src_pad_var": [1, 1, 16],
//            "_permute": [0, 2, 1, 3],
//            "_src_fuse": [0, 1, 3],
//            "_pattern": "Transdata",
//            "_common_info": [0, 16, 32, 0, 0],
//            "_ub_info": [[65280, 32640], [65280], [65280]],
//            "_bh_x1x0": [1, 2],
//            "_bh_c1c0": [3],
//            "_bh_permute": [0, 3, 1, 2],
//            "_bn_x1x0": [0, 1],
//            "_bn_c1c0": [3],
//            "_bn_permute": [0, 3, 1, 2]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 0;
  op_compile_info.align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad_mode = {0, 0, 2};
  op_compile_info.src_pad_var = {1, 1, 16};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.ub_info = {{65280, 32640}, {65280}, {65280}};

  op_compile_info.bh_x1x0 = {1, 2};
  op_compile_info.bh_c1c0 = {3,};
  op_compile_info.bh_permute = {0, 3, 1, 2};
  op_compile_info.bn_x1x0 = {0, 1};
  op_compile_info.bn_c1c0 = {3,};
  op_compile_info.bn_permute = {0, 3, 1, 2};


  vector<vector<int64_t>> inputs {
          {2, 1, 1, 56, 16}
  };
  vector<vector<int64_t>> outputs {
          {2, 1, 56, 3}
  };

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataBorrow transdata(op_type, op_compile_info, runInfo, input, output);
  transdata.SetAttr(sch_branch, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NC1HWC0 -> NHWC, Fp32, DYNAMIC
TEST_F(TransdataTiling, Backward_5HD_NHWC_Fp32_Dynamic_BH) {
  //    string compileInfo = R"(
  //          { "_src_pad_mode": [0, 0, 2],
  //            "_src_pad_var": [1, 1, 16],
  //            "_permute": [0, 2, 1, 3],
  //            "_src_fuse": [0, 1, 3],
  //            "_pattern": "Transdata",
  //            "_common_info": [0, 8, 32, 0, 0],
  //            "_ub_info": [[32640, 16256], [65280], [65280]],
  //            "_bh_x1x0": [1, 2],
  //            "_bh_c1c0": [3],
  //            "_bh_permute": [0, 4, 1, 2, 3],
  //            "_bn_x1x0": [0, 1],
  //            "_bn_c1c0": [3],
  //            "_bn_permute": [0, 4, 1, 2, 3]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 0;
  op_compile_info.align_size = 8;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad_mode = {0, 0, 2};
  op_compile_info.src_pad_var = {1, 1, 16};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.ub_info = {{32640, 16256}, {65280}, {65280}};

  op_compile_info.bh_x1x0 = {1, 2};
  op_compile_info.bh_c1c0 = {3,};
  op_compile_info.bh_permute = {0, 4, 1, 2, 3};
  op_compile_info.bn_x1x0 = {0, 1};
  op_compile_info.bn_c1c0 = {3,};
  op_compile_info.bn_permute = {0, 4, 1, 2, 3};


  vector<vector<int64_t>> inputs {
          {2, 2, 20, 56, 16}
  };
  vector<vector<int64_t>> outputs {
          {2, 20, 56, 21}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataBorrow transdata(op_type, op_compile_info, runInfo, input, output);
  transdata.SetAttr(sch_branch, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}

// Test Case: NC1HWC0 -> NHWC, Fp32, Const Compile
TEST_F(TransdataTiling, Backward_5HD_NHWC_Fp32_Compile) {
  //    string compileInfo = R"(
  //          { "_src_pad_mode": [0, 0, 2, 0],
  //            "_src_pad_var": [1, 1, 16, 1],
  //            "_permute": [0, 2, 1, 3, 4],
  //            "_src_fuse": [0, 1, 3],
  //            "_pattern": "Transdata",
  //            "_common_info": [0, 16, 32, 1, 1],
  //            "_ub_info": [[-1], [-1], [65280]],
  //            "_bh_x1x0": [1, 2],
  //            "_bh_c1c0": [3],
  //            "_bh_permute": [0, 4, 1, 2, 3]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 0;
  op_compile_info.align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 1;
  op_compile_info.is_const_compile = 1;

  op_compile_info.src_pad_mode = {0, 0, 2, 0};
  op_compile_info.src_pad_var = {1, 1, 16, 1};
  op_compile_info.permute = {0, 2, 1, 3, 4};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.ub_info = {{-1}, {-1}, {65280}};
  op_compile_info.bh_x1x0 = {1, 2};
  op_compile_info.bh_c1c0 = {3,};
  op_compile_info.bh_permute = {0, 4, 1, 2, 3};


  vector<vector<int64_t>> inputs {
          {2, 2, 1120, 16, 2}
  };
  vector<vector<int64_t>> outputs {
          {2, 1120, 21, 2}
  };

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (std::size_t i = 0; i < inputs.size(); i++) {
  contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
  contruct_tensor(op_desc, outputs[i], dtype, false);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;
  // main process
  Shape input;
  Shape output;
  Shape reshape;
  TransdataClassify classify(op_paras, op_compile_info);
  classify.GetInputOutput(input, output, reshape);
  size_t sch_branch = classify.ChooseStrategy(input, output);
  size_t transpose_work = classify.TransposeWork(input, output);
  string op_type = "Transdata";
  TransdataBorrow transdata(op_type, op_compile_info, runInfo, input, output);
  transdata.SetAttr(sch_branch, transpose_work);
  ASSERT_TRUE(transdata.DoTiling());
}


// Test Case: mock transdata_dsl.cc
TEST_F(TransdataTiling, Mock_Transdata_DSL) {
    string compileInfo = R"(
          { "_src_pad_mode": [0, 0, 2],
            "_src_pad_var": [1, 1, 16],
            "_permute": [0, 2, 1, 3],
            "_src_fuse": [0, 1, 3],
            "_pattern": "Transdata",
            "_common_info": [0, 8, 32, 0, 0],
            "_ub_info": [[32640, 16256], [65280], [65280]],
            "_bh_x1x0": [1, 2],
            "_bh_c1c0": [3],
            "_bh_permute": [0, 4, 1, 2, 3],
            "_bn_x1x0": [0, 1],
            "_bn_c1c0": [3],
            "_bn_permute": [0, 4, 1, 2, 3]})";
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  CompileInfoTransdataDSL info("transdata", parsed_compile_info);
  ASSERT_TRUE(true);
}