/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "max_pool_v3.h"
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "tests/depends/faker/kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "runtime/kernel_context.h"
#include "runtime/tiling_data.h"

namespace gert_test {
class MaxPoolV3UT : public testing::Test {};
TEST_F(MaxPoolV3UT, TilingOk) {
  gert::StorageShape x_shape = {{1, 4, 56, 56, 16}, {1, 4, 56, 56, 16}};

  std::vector<gert::StorageShape> output_shapes(1);
  std::vector<gert::StorageShape *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }
  // compile info
  int32_t ub_ele = 126976;
  int32_t core_num = 32;
  int32_t ksize_h = 3;
  int32_t ksize_w = 3;
  int32_t strides_h = 2;
  int32_t strides_w = 2;
  int32_t padding = 2;    // SAME
  int32_t ceil_mode = 0;  // floor
  int32_t pad_top = 1;
  int32_t pad_bottom = 1;
  int32_t pad_left = 1;
  int32_t pad_right = 1;
  int32_t global = 0;
  optiling::MaxPoolV3CompileInfo compile_info;
  compile_info.core_num = core_num;
  compile_info.ksize_h = ksize_h;
  compile_info.ksize_w = ksize_w;
  compile_info.strides_h = strides_h;
  compile_info.strides_w = strides_w;
  compile_info.padding = padding;
  compile_info.ceil_mode = ceil_mode;
  compile_info.pad_top = pad_top;
  compile_info.pad_bottom = pad_bottom;
  compile_info.pad_left = pad_left;
  compile_info.pad_right = pad_right;
  compile_info.global = global;
  compile_info.ub_ele = ub_ele;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&x_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0)
                    .TilingData(param.get())
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
  // todo check tiling result
}

TEST_F(MaxPoolV3UT, TilingParseOk) {
  char *json_str = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 2, \"ceil_mode\": 0, \"pad_top\": 1, \"pad_bottom\": 1, \"pad_left\": 1, "
      "\"pad_right\": 1, \"global\": 0}}";
  optiling::MaxPoolV3CompileInfo compile_info;
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({json_str})
                    .Outputs({&compile_info})
                    .IrInstanceNum({1})
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3"), nullptr);
  auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->tiling_parse;
  ASSERT_NE(tiling_prepare_func, nullptr);

  EXPECT_EQ(tiling_prepare_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  int32_t ub_ele = 126976;
  int32_t core_num = 32;
  int32_t ksize_h = 3;
  int32_t ksize_w = 3;
  int32_t strides_h = 2;
  int32_t strides_w = 2;
  int32_t padding = 2;    // SAME
  int32_t ceil_mode = 0;  // floor
  int32_t pad_top = 1;
  int32_t pad_bottom = 1;
  int32_t pad_left = 1;
  int32_t pad_right = 1;
  int32_t global = 0;
  EXPECT_EQ(compile_info.ub_ele, ub_ele);
  EXPECT_EQ(compile_info.core_num, core_num);
  EXPECT_EQ(compile_info.ksize_h, ksize_h);
  EXPECT_EQ(compile_info.ksize_w, ksize_w);
  EXPECT_EQ(compile_info.strides_h, strides_h);
  EXPECT_EQ(compile_info.strides_w, strides_w);
  EXPECT_EQ(compile_info.padding, padding);
  EXPECT_EQ(compile_info.ceil_mode, ceil_mode);
  EXPECT_EQ(compile_info.pad_top, pad_top);
  EXPECT_EQ(compile_info.pad_bottom, pad_bottom);
  EXPECT_EQ(compile_info.pad_left, pad_left);
  EXPECT_EQ(compile_info.pad_right, pad_right);
  EXPECT_EQ(compile_info.global, global);
}

TEST_F(MaxPoolV3UT, DetaDepencyOk) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3"), nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->IsInputDataDependency(0));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->IsInputDataDependency(1));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->IsInputDataDependency(2));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->IsInputDataDependency(3));
}
}  // namespace gert_test