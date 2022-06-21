/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "gtest/gtest.h" 
#ifndef private
#define private public
#define protected public
#endif      
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "aicpu_read_file.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
#include "iostream"

using namespace std;
using namespace aicpu;

class TEST_GRIDSAMPLER3DGRAD_UT : public testing::Test{};

#define CREATE_NODEDEF(shapes, data_types, datas, attr1, attr2, attr3)     \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(),"GridSampler3DGrad", "GridSampler3DGrad")  \
      .Input({"grad", data_types[0], shapes[0], datas[0]})                 \
      .Input({"x", data_types[1], shapes[1], datas[1]})                    \
      .Input({"grid", data_types[2], shapes[2], datas[2]})                 \
      .Output({"dx", data_types[3], shapes[3], datas[3]})                  \
      .Output({"dgrid", data_types[4], shapes[4], datas[4]})               \
      .Attr("interpolation_mode", attr1)                                   \
      .Attr("padding_mode", attr2)                                         \
      .Attr("align_corners", attr3);

template <typename T>
T clip_coordinates_set_grad(T x, int64_t clip_limit, T *grad_x) {
  if (x <= static_cast<T>(0)) {
    *grad_x = static_cast<T>(0);
    return static_cast<T>(0);
  } else {
    T max = static_cast<T>(clip_limit - 1);
    if (x >= max) {
      *grad_x = static_cast<T>(0);
      return max;
    } else {
      *grad_x = static_cast<T>(1);
      return x;
    }
  }
}

template <typename T>
T reflect_coordinates_set_grad(T x, int64_t twice_low, int64_t twice_high, T *grad_x) {
  if (twice_low == twice_high) {
    *grad_x = static_cast<T>(0);
    return static_cast<T>(0);
  }
  int64_t grad_x_mult_;
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  x = x - min;
  if (x < static_cast<T>(0)){
    grad_x_mult_ = -1;
    x = -x;
  } else {
    grad_x_mult_ = 1;
  }
  T extra = std::fmod(x, span);
  int64_t flips = static_cast<int64_t>(std::floor(x / span));
  if (flips % 2 == 0) {
    *grad_x = static_cast<T>(grad_x_mult_);
    return extra + min;
  } else {
    *grad_x = static_cast<T>(-grad_x_mult_);
    return span - extra + min;
  }
}

bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

template <typename T>
void safe_add_3d(T *data, int64_t d, int64_t h, int64_t w,
    int64_t sD, int64_t sH, int64_t sW, int64_t D, int64_t H, int64_t W, T delta) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    data[d * sD + h * sH + w * sW] += delta;
  }
}

template <typename T>
T grid_sampler_compute_source_index_set_grad(
    T coord, int64_t size, std::string padding_mode, bool align_corners, T *grad_x) {
  T grad_clip, grad_refl;
  if (align_corners) {
    *grad_x = static_cast<T>(size - 1) / 2;
    coord = ((coord + 1) / 2) * (size - 1);
  } else {
    *grad_x = static_cast<T>(size) / 2;
    coord = ((coord + 1) * size - 1) / 2;
  }
  if (padding_mode == "border") {
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_x = (*grad_x) * grad_clip;
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, 2 * (size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2 * size - 1, &grad_refl);
    }
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_x = (*grad_x) * grad_refl * grad_clip;
  }
  return coord;
}

void CalcExpectWithHalfData(const NodeDef &node_def, Eigen::half expect_dx[], Eigen::half expect_dgrid[]) {
  auto grad_data = node_def.MutableInputs(0);
  auto grad_data_addr = reinterpret_cast<Eigen::half *>(grad_data->GetData());
  auto grad_shape = grad_data->GetTensorShape();
  auto grad_dims = grad_shape->GetDimSizes();
  int64_t grad_stride[5];
  int64_t stride_tmp = 1;
  for (int32_t i = 4; i > -1; i--) {
    grad_stride[i] = stride_tmp;
    stride_tmp *= grad_dims[i];
  }
  auto x_data = node_def.MutableInputs(1);
  auto x_data_addr = reinterpret_cast<Eigen::half *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  int64_t x_stride[5];
  stride_tmp = 1;
  for (int32_t i = 4; i > -1; i--) {
    x_stride[i] = stride_tmp;
    stride_tmp *= x_dims[i];
  }
  auto grid_data = node_def.MutableInputs(2);
  auto grid_data_addr = reinterpret_cast<Eigen::half *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  uint64_t grid_data_size = grid_data->GetDataSize();
  int64_t grid_stride[5];
  stride_tmp = 1;
  for (int32_t i = 4; i > -1; i--) {
    grid_stride[i] = stride_tmp;
    stride_tmp *= grid_dims[i];
  }
  auto attrs = node_def.Attrs();
  auto attr1 = attrs["interpolation_mode"];
  auto attr2 = attrs["padding_mode"];
  auto attr3 = attrs["align_corners"];
  std::string interpolation_mode = attr1->GetString();
  std::string padding_mode = attr2->GetString();
  bool align_corners = attr3->GetBool();
  for (int64_t n = 0; n < x_dims[0]; n++){
    int64_t x_ptr_N = n * x_stride[0];
    int64_t grid_ptr_N = n * grid_stride[0];
    int64_t dgrid_ptr_NDHW = n * grid_stride[0];
    for (int64_t d = 0; d < grid_dims[1]; d++) {
      for (int64_t h = 0; h < grid_dims[2]; h++) {
        for (int64_t w = 0; w < grid_dims[3]; w++, dgrid_ptr_NDHW += grid_stride[3]) {
          int64_t grid_ptr_NDHW = grid_ptr_N + d * grid_stride[1] + h * grid_stride[2] + w * grid_stride[3];
          float x = static_cast<float>(grid_data_addr[grid_ptr_NDHW]);
          float y = static_cast<float>(grid_data_addr[grid_ptr_NDHW + grid_stride[4]]);
          float z = static_cast<float>(grid_data_addr[grid_ptr_NDHW + 2 * grid_stride[4]]);
          float gx_mult, gy_mult, gz_mult;
          x = grid_sampler_compute_source_index_set_grad(x, x_dims[4], padding_mode, align_corners, &gx_mult);
          y = grid_sampler_compute_source_index_set_grad(y, x_dims[3], padding_mode, align_corners, &gy_mult);
          z = grid_sampler_compute_source_index_set_grad(z, x_dims[2], padding_mode, align_corners, &gz_mult);
          if (interpolation_mode == "bilinear") {
            int64_t x_tnw = static_cast<int64_t>(std::floor(x));
            int64_t y_tnw = static_cast<int64_t>(std::floor(y));
            int64_t z_tnw = static_cast<int64_t>(std::floor(z));
            int64_t x_tne = x_tnw + 1;
            int64_t y_tne = y_tnw;
            int64_t z_tne = z_tnw;
            int64_t x_tsw = x_tnw;
            int64_t y_tsw = y_tnw + 1;
            int64_t z_tsw = z_tnw;
            int64_t x_tse = x_tnw + 1;
            int64_t y_tse = y_tnw + 1;
            int64_t z_tse = z_tnw;
            int64_t x_bnw = x_tnw;
            int64_t y_bnw = y_tnw;
            int64_t z_bnw = z_tnw + 1;
            int64_t x_bne = x_tnw + 1;
            int64_t y_bne = y_tnw;
            int64_t z_bne = z_tnw + 1;
            int64_t x_bsw = x_tnw;
            int64_t y_bsw = y_tnw + 1;
            int64_t z_bsw = z_tnw + 1;
            int64_t x_bse = x_tnw + 1;
            int64_t y_bse = y_tnw + 1;
            int64_t z_bse = z_tnw + 1;
            Eigen::half tnw = static_cast<Eigen::half>((x_bse - x) * (y_bse - y) * (z_bse - z));
            Eigen::half tne = static_cast<Eigen::half>((x - x_bsw) * (y_bsw - y) * (z_bsw - z));
            Eigen::half tsw = static_cast<Eigen::half>((x_bne - x) * (y - y_bne) * (z_bne - z));
            Eigen::half tse = static_cast<Eigen::half>((x - x_bnw) * (y - y_bnw) * (z_bnw - z));
            Eigen::half bnw = static_cast<Eigen::half>((x_tse - x) * (y_tse - y) * (z - z_tse));
            Eigen::half bne = static_cast<Eigen::half>((x - x_tsw) * (y_tsw - y) * (z - z_tsw));
            Eigen::half bsw = static_cast<Eigen::half>((x_tne - x) * (y - y_tne) * (z - z_tne));
            Eigen::half bse = static_cast<Eigen::half>((x - x_tnw) * (y - y_tnw) * (z - z_tnw));
            Eigen::half gx = static_cast<Eigen::half>(0), gy = static_cast<Eigen::half>(0), gz = static_cast<Eigen::half>(0);
            int64_t grad_ptr_NCDHW = n * grad_stride[0] +
                                  d * grad_stride[2] +
                                  h * grad_stride[3] +
                                  w * grad_stride[4] ;
            int64_t dx_ptr_NC = n * x_stride[0];
            int64_t x_ptr_NC = x_ptr_N;
            for (int64_t c = 0; c < x_dims[1]; c++, grad_ptr_NCDHW += grad_stride[1], x_ptr_NC += x_stride[1], dx_ptr_NC += x_stride[1]) {
              Eigen::half grad_out = grad_data_addr[grad_ptr_NCDHW];
              safe_add_3d(&expect_dx[dx_ptr_NC], z_tnw, y_tnw, x_tnw, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], tnw * grad_out);
              safe_add_3d(&expect_dx[dx_ptr_NC], z_tne, y_tne, x_tne, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], tne * grad_out);
              safe_add_3d(&expect_dx[dx_ptr_NC], z_tsw, y_tsw, x_tsw, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], tsw * grad_out);
              safe_add_3d(&expect_dx[dx_ptr_NC], z_tse, y_tse, x_tse, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], tse * grad_out);
              safe_add_3d(&expect_dx[dx_ptr_NC], z_bnw, y_bnw, x_bnw, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], bnw * grad_out);
              safe_add_3d(&expect_dx[dx_ptr_NC], z_bne, y_bne, x_bne, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], bne * grad_out);
              safe_add_3d(&expect_dx[dx_ptr_NC], z_bsw, y_bsw, x_bsw, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], bsw * grad_out);
              safe_add_3d(&expect_dx[dx_ptr_NC], z_bse, y_bse, x_bse, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], bse * grad_out);
              if (within_bounds_3d(z_tnw, y_tnw, x_tnw, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_tnw * x_stride[2] + y_tnw * x_stride[3] + x_tnw * x_stride[4];
                Eigen::half tnw_val = x_data_addr[x_index];
                gx -= tnw_val * static_cast<Eigen::half>((y_bse - y) * (z_bse - z)) * grad_out;
                gy -= tnw_val * static_cast<Eigen::half>((x_bse - x) * (z_bse - z)) * grad_out;
                gz -= tnw_val * static_cast<Eigen::half>((x_bse - x) * (y_bse - y)) * grad_out; 
              }
              if (within_bounds_3d(z_tne, y_tne, x_tne, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_tne * x_stride[2] + y_tne * x_stride[3] + x_tne * x_stride[4];
                Eigen::half tne_val = x_data_addr[x_index];
                gx += tne_val * static_cast<Eigen::half>((y_bsw - y) * (z_bsw - z)) * grad_out;
                gy -= tne_val * static_cast<Eigen::half>((x - x_bsw) * (z_bsw - z)) * grad_out;
                gz -= tne_val * static_cast<Eigen::half>((x - x_bsw) * (y_bsw - y)) * grad_out;
              }
              if (within_bounds_3d(z_tsw, y_tsw, x_tsw, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_tsw * x_stride[2] + y_tsw * x_stride[3] + x_tsw * x_stride[4];
                Eigen::half tsw_val = x_data_addr[x_index];
                gx -= tsw_val * static_cast<Eigen::half>((y - y_bne) * (z_bne - z)) * grad_out;
                gy += tsw_val * static_cast<Eigen::half>((x_bne - x) * (z_bne - z)) * grad_out;
                gz -= tsw_val * static_cast<Eigen::half>((x_bne - x) * (y - y_bne)) * grad_out;
              }
              if (within_bounds_3d(z_tse, y_tse, x_tse, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_tse * x_stride[2] + y_tse * x_stride[3] + x_tse * x_stride[4];
                Eigen::half tse_val = x_data_addr[x_index];
                gx += tse_val * static_cast<Eigen::half>((y - y_bnw) * (z_bnw - z)) * grad_out;
                gy += tse_val * static_cast<Eigen::half>((x - x_bnw) * (z_bnw - z)) * grad_out;
                gz -= tse_val * static_cast<Eigen::half>((x - x_bnw) * (y - y_bnw)) * grad_out;
              }
              if (within_bounds_3d(z_bnw, y_bnw, x_bnw, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_bnw * x_stride[2] + y_bnw * x_stride[3] + x_bnw * x_stride[4];
                Eigen::half bnw_val = x_data_addr[x_index];
                gx -= bnw_val * static_cast<Eigen::half>((y_tse - y) * (z - z_tse)) * grad_out;
                gy -= bnw_val * static_cast<Eigen::half>((x_tse - x) * (z - z_tse)) * grad_out;
                gz += bnw_val * static_cast<Eigen::half>((x_tse - x) * (y_tse - y)) * grad_out;
              }
              if (within_bounds_3d(z_bne, y_bne, x_bne, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_bne * x_stride[2] + y_bne * x_stride[3] + x_bne * x_stride[4];
                Eigen::half bne_val = x_data_addr[x_index];
                gx += bne_val * static_cast<Eigen::half>((y_tsw - y) * (z - z_tsw)) * grad_out;
                gy -= bne_val * static_cast<Eigen::half>((x - x_tsw) * (z - z_tsw)) * grad_out;
                gz += bne_val * static_cast<Eigen::half>((x - x_tsw) * (y_tsw - y)) * grad_out;
              }
              if (within_bounds_3d(z_bsw, y_bsw, x_bsw, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_bsw * x_stride[2] + y_bsw * x_stride[3] + x_bsw * x_stride[4];
                Eigen::half bsw_val = x_data_addr[x_index];
                gx -= bsw_val * static_cast<Eigen::half>((y - y_tne) * (z - z_tne)) * grad_out;
                gy += bsw_val * static_cast<Eigen::half>((x_tne - x) * (z - z_tne)) * grad_out;
                gz += bsw_val * static_cast<Eigen::half>((x_tne - x) * (y - y_tne)) * grad_out;
              }
              if (within_bounds_3d(z_bse, y_bse, x_bse, x_dims[2], x_dims[3], x_dims[4])) {
                auto x_index = x_ptr_NC + z_bse * x_stride[2] + y_bse * x_stride[3] + x_bse * x_stride[4];
                Eigen::half bse_val = x_data_addr[x_index];
                gx += bse_val * static_cast<Eigen::half>((y - y_tnw) * (z - z_tnw)) * grad_out;
                gy += bse_val * static_cast<Eigen::half>((x - x_tnw) * (z - z_tnw)) * grad_out;
                gz += bse_val * static_cast<Eigen::half>((x - x_tnw) * (y - y_tnw)) * grad_out;
              }
            }
            expect_dgrid[dgrid_ptr_NDHW] = static_cast<Eigen::half>(gx_mult) * gx;
            expect_dgrid[dgrid_ptr_NDHW + 1] = static_cast<Eigen::half>(gy_mult) * gy;
            expect_dgrid[dgrid_ptr_NDHW + 2] = static_cast<Eigen::half>(gz_mult) * gz;
          } else if (interpolation_mode == "nearest") {
            int64_t x_nearest = static_cast<int64_t>(std::round(x));
            int64_t y_nearest = static_cast<int64_t>(std::round(y));
            int64_t z_nearest = static_cast<int64_t>(std::round(z));
            int64_t grad_ptr_NCDHW = n * grad_stride[0] +
                                  d * grad_stride[2] +
                                  h * grad_stride[3] +
                                  w * grad_stride[4] ;
            int64_t dx_ptr_NC = n * x_stride[0];
            for (int64_t c = 0; c < x_dims[1]; c++, grad_ptr_NCDHW += grad_stride[1], dx_ptr_NC += x_stride[1]) {
              safe_add_3d(&expect_dx[dx_ptr_NC], z_nearest, y_nearest, x_nearest, x_stride[2], x_stride[3], 
                          x_stride[4], x_dims[2], x_dims[3], x_dims[4], grad_data_addr[grad_ptr_NCDHW]);
            }
            expect_dgrid[dgrid_ptr_NDHW] = static_cast<Eigen::half>(0);
            expect_dgrid[dgrid_ptr_NDHW + 1] = static_cast<Eigen::half>(0);
            expect_dgrid[dgrid_ptr_NDHW + 2] = static_cast<Eigen::half>(0);
          }
        }
      }
    }
  }
}

// read input and output data from files which generate by your python file
template<typename T>
void RunGridSampler3DGradKernel1(vector<string> data_files,
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes,
                   string attr1 = "bilinear",
                   string attr2 = "zeros",
                   bool attr3 = false) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T *input1 = new T[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T *input2 = new T[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  // read data from file for input3
  data_path = ktestcaseFilePath + data_files[2];
  uint64_t input3_size = CalTotalElements(shapes, 2);
  T *input3 = new T[input3_size];
  status = ReadFile(data_path, input3, input3_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 3);
  T *output1 = new T[output1_size];
  uint64_t output2_size = CalTotalElements(shapes, 4);
  T *output2 = new T[output2_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)input3,
                          (void *)output1,
                          (void *)output2};

  CREATE_NODEDEF(shapes, data_types, datas, attr1, attr2, attr3);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[3];
  T *output_exp1 = new T[output1_size];
  status = ReadFile(data_path, output_exp1, output1_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + data_files[4];
  T *output_exp2 = new T[output2_size];
  status = ReadFile(data_path, output_exp2, output2_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output1, output_exp1, output1_size);
  EXPECT_EQ(compare, true);
  compare = CompareResult(output2, output_exp2, output2_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] input3;
  delete [] output1;
  delete [] output2;
  delete [] output_exp1;
  delete [] output_exp2;
}

void RunGridSampler3DGradKernel2(vector<DataType> data_types,
                             vector<vector<int64_t>> &shapes,
                             string attr1 = "bilinear",
                             string attr2 = "zeros",
                             bool attr3 = false){
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  Eigen::half *input1 = new Eigen::half[input1_size];
  SetRandomValue<Eigen::half>(input1, input1_size, -10, 10);

  // gen data use SetRandomValue for input2
  uint64_t input2_size = CalTotalElements(shapes, 1);
  Eigen::half *input2 = new Eigen::half[input2_size];
  SetRandomValue<Eigen::half>(input2, input2_size, 0, 255);
  
  // gen data use SetRandomValue for input3
  uint64_t input3_size = CalTotalElements(shapes, 2);
  Eigen::half *input3 = new Eigen::half[input3_size];
  SetRandomValue<Eigen::half>(input3, input3_size, static_cast<float>(-1.0), static_cast<float>(1.0));

  uint64_t output1_size = CalTotalElements(shapes, 3);
  Eigen::half *output1 = new Eigen::half[output1_size];
  uint64_t output2_size = CalTotalElements(shapes, 4);
  Eigen::half *output2 = new Eigen::half[output2_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)input3,
                          (void *)output1,
                          (void *)output2};

  CREATE_NODEDEF(shapes, data_types, datas, attr1, attr2, attr3);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  Eigen::half *output_exp1 = new Eigen::half[output1_size];
  Eigen::half *output_exp2 = new Eigen::half[output2_size];
  CalcExpectWithHalfData(*node_def.get(), output_exp1, output_exp2);

  bool compare = CompareResult(output1, output_exp1, output1_size);
  EXPECT_EQ(compare, true);
  compare = CompareResult(output2, output_exp2, output2_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] input3;
  delete [] output1;
  delete [] output2;
  delete [] output_exp1;    
  delete [] output_exp2;  
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD1_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 3, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 3, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_1.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_1.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_1.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_1.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_1.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "bilinear", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD2_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{2, 3, 2, 44, 44}, {2, 3, 3, 44, 44}, {2, 2, 44, 44, 3}, {2, 3, 3, 44, 44}, {2, 2, 44, 44, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_2.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_2.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_2.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_2.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_2.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "bilinear", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD3_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_3.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_3.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_3.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_3.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_3.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "bilinear", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD4_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{2, 2, 2, 5, 5}, {2, 2, 4, 7, 7}, {2, 2, 5, 5, 3}, {2, 2, 4, 7, 7}, {2, 2, 5, 5, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_4.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_4.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_4.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_4.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_4.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "bilinear", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD5_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{1, 3, 1, 3, 2}, {1, 3, 2, 4, 4}, {1, 1, 3, 2, 3}, {1, 3, 2, 4, 4}, {1, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_5.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_5.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_5.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_5.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_5.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "bilinear", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD6_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_6.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_6.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_6.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_6.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_6.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "bilinear", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD7_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_7.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_7.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_7.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_7.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_7.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "bilinear", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD8_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_8.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_8.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_8.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_8.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_8.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "bilinear", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD9_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_9.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_9.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_9.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_9.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_9.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "bilinear", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD10_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{7, 1, 4, 3, 2}, {7, 1, 7, 7, 7}, {7, 4, 3, 2, 3}, {7, 1, 7, 7, 7}, {7, 4, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_10.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_10.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_10.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_10.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_10.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "bilinear", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD11_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_11.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_11.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_11.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_11.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_11.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "bilinear", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD12_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_12.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_12.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_12.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_12.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_12.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "bilinear", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD13_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 3, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 3, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_13.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_13.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_13.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_13.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_13.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "nearest", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD14_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{2, 3, 2, 44, 44}, {2, 3, 3, 44, 44}, {2, 2, 44, 44, 3}, {2, 3, 3, 44, 44}, {2, 2, 44, 44, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_14.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_14.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_14.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_14.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_14.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "nearest", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD15_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_15.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_15.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_15.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_15.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_15.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "nearest", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD16_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{2, 2, 2, 5, 5}, {2, 2, 4, 7, 7}, {2, 2, 5, 5, 3}, {2, 2, 4, 7, 7}, {2, 2, 5, 5, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_16.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_16.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_16.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_16.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_16.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "nearest", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD17_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{1, 3, 1, 3, 2}, {1, 3, 2, 4, 4}, {1, 1, 3, 2, 3}, {1, 3, 2, 4, 4}, {1, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_17.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_17.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_17.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_17.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_17.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "nearest", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD18_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_18.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_18.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_18.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_18.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_18.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "nearest", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD19_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_19.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_19.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_19.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_19.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_19.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "nearest", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD20_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_20.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_20.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_20.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_20.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_20.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "nearest", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD21_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_21.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_21.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_21.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_21.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_21.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "nearest", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD22_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{7, 1, 4, 3, 2}, {7, 1, 7, 7, 7}, {7, 4, 3, 2, 3}, {7, 1, 7, 7, 7}, {7, 4, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_22.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_22.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_22.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_22.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_22.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "nearest", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD23_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_23.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_23.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_23.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_23.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_23.txt"};
  RunGridSampler3DGradKernel1<double>(files, data_types, shapes, "nearest", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD24_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 1, 3, 2}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}};
  vector<string>files{"grid_sampler_3d_grad/data/gridsampler3dgrad_data_input1_24.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input2_24.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_input3_24.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output1_24.txt",
                      "grid_sampler_3d_grad/data/gridsampler3dgrad_data_output2_24.txt"};
  RunGridSampler3DGradKernel1<float>(files, data_types, shapes, "nearest", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD25_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 3, 2}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "bilinear", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD26_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 8, 8}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "bilinear", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD27_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{2, 4, 1, 8, 8}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "bilinear", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD28_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 3, 2}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "nearest", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD29_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 8, 8}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "nearest", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD30_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{2, 4, 1, 8, 8}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "nearest", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD31_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 3, 2}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "bilinear", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD32_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 8, 8}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "bilinear", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD33_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{2, 4, 1, 8, 8}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "bilinear", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD34_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 3, 2}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}, {32, 3, 2, 8, 5}, {32, 1, 3, 2, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "nearest", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD35_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 3, 1, 8, 8}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}, {32, 3, 2, 14, 14}, {32, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "nearest", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, GRIDSAMPLER3DGRAD36_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{2, 4, 1, 8, 8}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}, {2, 4, 2, 14, 14}, {2, 1, 8, 8, 3}};
  RunGridSampler3DGradKernel2(data_types, shapes, "nearest", "reflection", false);
}

// exception instance
TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 4, 1, 8, 8}, {2, 4, 2, 3, 3}, {2, 1, 8, 8, 3}, {2, 4, 2, 3, 3}, {2, 1, 8, 8, 3}};
  float output1[144] = {(float)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)nullptr, (void *)output1, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas,  "bilinear", "zeros", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, INPUT_TYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 4, 1, 1, 1}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}};
  float input1[8] = {(float)0};
  Eigen::half input2[16] = {static_cast<Eigen::half>(0)};
  float input3[6] = {(float)0};
  float output1[16] = {(float)0};
  float output2[6] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas,  "bilinear", "border", false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, INPUT_TYPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 4, 1, 1, 1}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}};
  int32_t input1[8] = {0};
  int32_t input2[16] = {0};
  int32_t input3[6] = {0};
  float output1[16] = {(float)0};
  float output2[6] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas,  "nearest", "zeros", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, ATTR_ERROR_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 4, 1, 1, 1}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}};
  float input1[8] = {(float)0};
  float input2[16] = {(float)0};
  float input3[6] = {(float)0};
  float output1[16] = {(float)0};
  float output2[6] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas,  "ERROR", "border", false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3DGRAD_UT, SHAPE_ERROR_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4, 1, 1, 1}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}, {2, 4, 2, 1, 1}, {2, 1, 1, 1, 3}};
  float input1[16] = {(float)0};
  float input2[16] = {(float)0};
  float input3[6] = {(float)0};
  float output1[16] = {(float)0};
  float output2[6] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas,  "nearest", "border", false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}