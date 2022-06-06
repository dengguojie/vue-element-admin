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
#include <gtest/gtest.h>
#include "strided_slice_v3.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

class StridedSliceV3UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_v3_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stried_slice_v3_tiling TearDown" << std::endl;
  }
  template <typename T>
  std::unique_ptr<uint8_t[]> ConstructConstTensor(const std::vector<T> &const_value) {
    auto const_dtype = ge::DT_INT32;
    if (sizeof(T) == sizeof(int64_t)) {
      const_dtype = ge::DT_INT64;
    }

    int64_t value_size = const_value.size();
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * value_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
    *input_tensor = {
        {{value_size}, {value_size}},        // storage shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // storage format
        gert::kFollowing,                    // placement
        const_dtype,                         // data type
        0,                                   // address
    };
    auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
    for (size_t j = 0; j < value_size; j++) {
      tensor_data[j] = const_value[j];
    }
    input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
    return input_tensor_holder;
  }

  template <typename T>
  void ConstructContextFaker(gert::StorageShape &x_shape, gert::StorageShape &y_shape,
                             std::map<std::string, std::vector<T>> &value_dict,
                             optiling::StridedSliceV3CompileInfo &compile_info, std::unique_ptr<uint8_t[]> &param) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->tiling;
    ASSERT_NE(tiling_func, nullptr);

    std::vector<T>& axes_value = value_dict["axis"];
    std::vector<T>& stride_value = value_dict["stride"];
    std::unique_ptr<uint8_t[]> begin_tensor_holder = ConstructConstTensor(value_dict["begin"]);
    std::unique_ptr<uint8_t[]> end_tensor_holder = ConstructConstTensor(value_dict["end"]);
    std::unique_ptr<uint8_t[]> axes_tensor_holder = ConstructConstTensor(axes_value);
    std::unique_ptr<uint8_t[]> stride_tensor_holder = ConstructConstTensor(stride_value);

    auto begin_tensor = reinterpret_cast<void *>(begin_tensor_holder.get());
    auto end_tensor = reinterpret_cast<void *>(end_tensor_holder.get());
    auto axes_tensor = reinterpret_cast<void *>(axes_tensor_holder.get());
    auto stride_tensor = reinterpret_cast<void *>(stride_tensor_holder.get());

    std::vector<uint32_t> instance_num = {1, 1, 1, 1, 1};
    if (axes_value.empty()) {
      instance_num[3] = 0;
    }
    if (stride_value.empty()) {
      instance_num[4] = 0;
    }
    auto holder = gert::TilingContextFaker()
                     .NodeIoNum(5, 1)
                     .IrInstanceNum(instance_num)
                     .InputShapes({&x_shape, begin_tensor, end_tensor, axes_tensor, stride_tensor})
                     .OutputShapes({&y_shape})
                     .CompileInfo(&compile_info)
                     .TilingData(param.get())
                     .Build();

    EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
    return;
  }
};

template <typename T>
static string to_string(void *buf, size_t size) {
  std::string result;
  const T *data = reinterpret_cast<const T *>(buf);
  size_t len = size / sizeof(T);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

TEST_F(StridedSliceV3UT, stried_slice_v3_normal) {
  optiling::StridedSliceV3CompileInfo compile_info;
  compile_info.block_dim = 32;
  compile_info.ub_size = 262144;
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {0, 0}},
      {"end", {9, 7}},
      {"axis", {2, 3}},
      {"stride", {1, 1}}};
  gert::StorageShape x_shape = {{5, 6, 17, 18}, {5, 6, 17, 18}};
  gert::StorageShape y_shape = {{5, 6, 9, 7}, {5, 6, 9, 7}};

  auto param = gert::TilingData::CreateCap(2048);
  ConstructContextFaker<int64_t>(x_shape, y_shape, value_dict, compile_info, param);
  gert::TilingData *tiling_data = reinterpret_cast<gert::TilingData *>(param.get());
  ASSERT_NE(tiling_data, nullptr);
  EXPECT_EQ(to_string<int64_t>(tiling_data->GetData(), tiling_data->GetDataSize()),
            "1 3 30 17 18 30 9 7 0 0 0 30 9 7 1 1 1 ");
}

TEST_F(StridedSliceV3UT, stried_slice_v3_paser_success) {
  optiling::StridedSliceV3CompileInfo compile_info;
  char* js_buf =
      "{\"vars\": {\"block_dim\": 32, \"begin_mask\": 0, \"end_mask\": 0, \"ellipsis_mask\": 0, \"new_axis_mask\": 0, "
      "\"shrink_axis_mask\": 0, \"ub_size\": 262144}}";
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({js_buf})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->tiling_parse;
  ASSERT_NE(tiling_prepare_func, nullptr);
  EXPECT_EQ(tiling_prepare_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}
