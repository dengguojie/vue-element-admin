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

#include <gtest/gtest.h>

#include "graph/utils/op_desc_utils.h"
#include "op_tiling/vector_tiling.h"
#include "op_tiling/transdata_dsl.h"
#include "op_tiling/transdata_dsl_general.h"
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

/* Test Case
 * **/
TEST_F(TransdataTiling, TransdataTiling0) {
  string compileInfo = R"(
    { "_pad_factor": 16,
      "_src_pad": [0, 2, 0],
      "_permute": [0, 1, 3, 2],
      "_src_fuse": [0, 1, 2],
      "_pattern": "Transdata",
      "_common_info": [1, 8, 16, 32, 0, 0],
      "_unknown_dims": [],
      "_ub_info": [[16256, 16256]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 8;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad = {0, 2, 0};
  op_compile_info.src_fuse = {0, 1, 2};
  op_compile_info.permute = {0, 1, 3, 2};
  op_compile_info.unknown_dims = {0, 1, 2};
  op_compile_info.ub_info = {{16256, 16256}};

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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}

TEST_F(TransdataTiling, TransdataTiling1) {
  string compileInfo = R"(
      { "_pad_factor": 16,
        "_src_pad": [0, 2, 0],
        "_permute": [0, 1, 3, 2],
        "_src_fuse": [0, 1, 2],
        "_pattern": "Transdata",
        "_common_info": [1, 16, 16, 32, 0, 0],
        "_unknown_dims": [],
        "_ub_info": [[16256, 16256]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 16;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad = {0, 2, 0};
  op_compile_info.src_fuse = {0, 1, 2};
  op_compile_info.permute = {0, 1, 3, 2};
  op_compile_info.unknown_dims = {0, 1, 2};
  op_compile_info.ub_info = {{16256, 16256}};

  vector<vector<int64_t>> inputs {
          {16, 192, 56, 56}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56, 56, 16}
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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}


TEST_F(TransdataTiling, TransdataTiling2) {

  string compileInfo = R"(
    { "_pad_factor": 16,
      "_src_pad": [0, 0, 2],
      "_permute": [0, 2, 1, 3],
      "_src_fuse": [0, 1, 3],
      "_pattern": "Transdata",
      "_common_info": [1, 8, 16, 32, 0, 0],
      "_unknown_dims": [],
      "_ub_info": [[16256, 16256]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 8;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad = {0, 0, 2};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.unknown_dims = {};
  op_compile_info.ub_info = {{16256, 16256}};

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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}

TEST_F(TransdataTiling, TransdataTiling3) {

  string compileInfo = R"(
      { "_pad_factor": 16,
        "_src_pad": [0, 0, 2],
        "_permute": [0, 2, 1, 3],
        "_src_fuse": [0, 1, 3],
        "_pattern": "Transdata",
        "_common_info": [1, 16, 16, 32, 0, 0],
        "_unknown_dims": [],
        "_ub_info": [[32000, 32000]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 16;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad = {0, 0, 2};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.unknown_dims = {};
  op_compile_info.ub_info = {{32000, 32000}};

  vector<vector<int64_t>> inputs {
          {16, 56, 56, 192}
  };
  vector<vector<int64_t>> outputs {
          {16, 12, 56, 56, 16}
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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}

TEST_F(TransdataTiling, TransdataTiling4) {
  string compileInfo = R"(
        { "_pad_factor": 16,
          "_src_pad": [0, 0, 2],
          "_permute": [0, 2, 1, 3],
          "_src_fuse": [0, 1, 3],
          "_pattern": "Transdata",
          "_common_info": [1, 16, 16, 32, 0, 0],
          "_unknown_dims": [],
          "_ub_info": [[32000, 32000]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 1;
  op_compile_info.align_size = 16;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad = {0, 0, 2};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.unknown_dims = {};
  op_compile_info.ub_info = {{32000, 32000}};

  vector<vector<int64_t>> inputs {
          {16, 56, 56, 21}
  };
  vector<vector<int64_t>> outputs {
          {16, 2, 56, 56, 16}
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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}

TEST_F(TransdataTiling, TransdataTiling5) {
  std::string compileInfo = R"(
      { "_pad_factor": 16,
        "_src_pad": [0, 2, 0],
        "_permute": [0, 1, 3, 2],
        "_src_fuse": [0, 1, 2],
        "_pattern": "Transdata",
        "_common_info": [0, 16, 16, 32, 0, 0],
        "_unknown_dims": [],
        "_ub_info": [[64000, 64000]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 0;
  op_compile_info.align_size = 16;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad = {0, 2, 0};
  op_compile_info.src_fuse = {0, 1, 2};
  op_compile_info.permute = {0, 1, 3, 2};
  op_compile_info.unknown_dims = {};
  op_compile_info.ub_info = {{64000, 64000}};

  vector<vector<int64_t>> inputs {
          {16, 12, 56, 56, 16}
  };
  vector<vector<int64_t>> outputs {
          {16, 192, 56, 56}
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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}

TEST_F(TransdataTiling, TransdataTiling6) {
  std::string compileInfo = R"(
          { "_pad_factor": 16,
            "_src_pad": [0, 1, 2],
            "_permute": [0, 2, 1, 3],
            "_src_fuse": [0, 1, 3],
            "_pattern": "Transdata",
            "_common_info": [0, 16, 16, 32, 0, 0],
            "_unknown_dims": [],
            "_ub_info": [[32000, 32000]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 0;
  op_compile_info.align_size = 16;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 0;
  op_compile_info.is_const_compile = 0;

  op_compile_info.src_pad = {0, 1, 2};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.unknown_dims = {};
  op_compile_info.ub_info = {{32000, 32000}};

  vector<vector<int64_t>> inputs {
          {16, 2, 56, 56, 16}
  };
  vector<vector<int64_t>> outputs {
          {16, 56, 56, 21}
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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}

TEST_F(TransdataTiling, TransdataTiling8) {
  std::string compileInfo = R"(
              { "_pad_factor": 16,
                "_src_pad": [0, 1, 2],
                "_permute": [0, 2, 1, 3],
                "_src_fuse": [0, 1, 3],
                "_pattern": "Transdata",
                "_common_info": [0, 16, 16, 32, 1, 1],
                "_unknown_dims": [],
                "_ub_info": [[32000, 32000]]})";

  CompileInfoTransdataDSL op_compile_info;
  op_compile_info.is_forward = 0;
  op_compile_info.align_size = 16;
  op_compile_info.pad_align_size = 16;
  op_compile_info.core_num = 32;
  op_compile_info.is_const = 1;
  op_compile_info.is_const_compile = 1;

  op_compile_info.src_pad = {0, 1, 2};
  op_compile_info.src_fuse = {0, 1, 3};
  op_compile_info.permute = {0, 2, 1, 3};
  op_compile_info.unknown_dims = {};
  op_compile_info.ub_info = {{32000, 32000}};

  vector<vector<int64_t>> inputs {
          {16, 2, 56, 56, 16}
  };
  vector<vector<int64_t>> outputs {
          {16, 56, 56, 21}
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
  int64_t sch_branch = classify.ChooseStrategy(input, output);
  TransdataGeneral transdata("Transdata", op_compile_info, runInfo, input, output, reshape);
  ASSERT_TRUE(transdata.DoTiling());
}