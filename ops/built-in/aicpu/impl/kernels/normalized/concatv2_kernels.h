/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of concatv2
 */

#ifndef _AICPU_CONCATV2_KERNELS_H_
#define _AICPU_CONCATV2_KERNELS_H_

#include <memory>
#include <vector>
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
template <typename T>
struct TTypes {
  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
      Matrix;

  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
      ConstMatrix;
};

class ConcatV2Kernel : public CpuKernel {
 public:
  ConcatV2Kernel()
      : data_type_(DT_DOUBLE),
        input_dims_(0),
        n_(0),
        output_concat_dim_(0),
        axis_(0),
        inputs_flat_dim0_(0) {}

  ~ConcatV2Kernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckAndInitParams(CpuKernelContext &ctx);

  template <typename T>
  uint32_t PrepareInput(
      CpuKernelContext &ctx,
      std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs) {
    inputs.reserve(n_);
    output_concat_dim_ = 0;
    auto input0_shape_ptr = ctx.Input(0)->GetTensorShape();
    for (uint32_t i = 0; i < n_; ++i) {
      Tensor *input_i_ptr = ctx.Input(i);
      KERNEL_CHECK_NULLPTR(input_i_ptr, KERNEL_STATUS_PARAM_INVALID,
                           "Get input[x%u] failed.", i);
      int64_t input_i_num = input_i_ptr->NumElements();
      if (input_i_num == 0) {
        continue;
      }
      auto input_i_shape_ptr = input_i_ptr->GetTensorShape();
      KERNEL_CHECK_NULLPTR(input_i_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                           "Get input[x%u] shape failed.", i);
      int32_t input_i_dims = input_i_shape_ptr->GetDims();
      KERNEL_CHECK_FALSE(
          (input_i_dims == input_dims_), KERNEL_STATUS_PARAM_INVALID,
          "Ranks of inputs should match: shape[0]=%d vs. shape[%u]=%d",
          input_dims_, i, input_i_dims);
      for (int32_t j = 0; j < input_dims_; ++j) {
        int64_t dim_ij = input_i_shape_ptr->GetDimSize(j);
        if (j == axis_) {
          output_concat_dim_ += input_i_dims > 0 ? dim_ij : 1;
          continue;
        }
        int64_t dim_0j = input0_shape_ptr->GetDimSize(j);
        KERNEL_CHECK_FALSE(
            (dim_0j == dim_ij), KERNEL_STATUS_PARAM_INVALID,
            "Dimensions of inputs should match: shape[0][%d]=%lld vs."
            "shape[%u][%d]=%lld",
            j, dim_0j, i, j, dim_ij);
      }

      int64_t inputs_flat_dim1 = input_i_num / inputs_flat_dim0_;
      auto input_i_data_ptr = input_i_ptr->GetData();
      KERNEL_CHECK_NULLPTR(input_i_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                           "Get input[x%u] data failed.", i);
      auto input_i = std::make_shared<typename TTypes<T>::ConstMatrix>(
          reinterpret_cast<T *>(input_i_data_ptr), inputs_flat_dim0_,
          inputs_flat_dim1);
      KERNEL_CHECK_NULLPTR(input_i, KERNEL_STATUS_PARAM_INVALID,
                           "Create input[x%u] failed!", i);
      inputs.emplace_back(std::move(input_i));
    }

    if (input_dims_ == 0) {
      output_concat_dim_ = n_;
    }
    return KERNEL_STATUS_OK;
  }

  template <typename T>
  uint32_t PrepareOutput(CpuKernelContext &ctx,
                         std::shared_ptr<typename TTypes<T>::Matrix> &output) {
    Tensor *output_ptr = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output failed.");
    auto output_data_ptr = output_ptr->GetData();
    KERNEL_CHECK_NULLPTR(output_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output data failed.");
    int64_t output_num = output_ptr->NumElements();
    int64_t output_dim1 = output_num / inputs_flat_dim0_;
    output = std::make_shared<typename TTypes<T>::Matrix>(
        reinterpret_cast<T *>(output_data_ptr), inputs_flat_dim0_, output_dim1);
    KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID,
                         "Create output matrix failed.");
    return KERNEL_STATUS_OK;
  }

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx) {
    std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>> inputs;
    KERNEL_CHECK_FALSE((PrepareInput<T>(ctx, inputs) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "PrepareInput failed.");
    std::shared_ptr<typename TTypes<T>::Matrix> output = nullptr;
    KERNEL_CHECK_FALSE((PrepareOutput<T>(ctx, output) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "PrepareOutput failed.");
    if (inputs.size() > 0) {
      return ConcatV2Compute<T>(ctx, inputs, output);
    }
    KERNEL_LOG_INFO("ConcatV2Kernel success.");
    return KERNEL_STATUS_OK;
  }

  template <typename T>
  uint32_t ConcatV2Compute(
      CpuKernelContext &ctx,
      const std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>>
          &inputs,
      std::shared_ptr<typename TTypes<T>::Matrix> &output) {
    size_t num_inputs = inputs.size();
    std::vector<ptrdiff_t> sizes;
    sizes.reserve(num_inputs);
    int64_t row_size = 0;
    for (const auto &input : inputs) {
      sizes.push_back(input->dimension(1));
      row_size += sizes.back();
    }
    uint32_t ret = KERNEL_STATUS_OK;
    auto work = [&row_size, &sizes, &inputs, &output, &num_inputs, &ret](
        int64_t start, int64_t end) {
      int64_t skipped_rows = start / row_size;
      T *out = output->data() + skipped_rows * row_size;
      T *out_start = output->data() + start;
      T *out_end = output->data() + end;

      // Handle partial row at start
      if (out < out_start) {
        for (size_t j = 0; j < num_inputs; ++j) {
          ptrdiff_t size = sizes[j];
          ptrdiff_t offset = out_start - out;
          if (size <= offset) {
            out += size;
            continue;
          }
          const T *inp = &(*inputs[j])(skipped_rows, 0);
          if (offset > 0) {
            out += offset;
            inp += offset;
            size -= offset;
          }
          size = std::min(size, out_end - out);
          KERNEL_CHECK_FALSE_EXEC((size > 0), break)
          size_t copy_size = size * sizeof(T);
          auto mem_ret = memcpy_s(out, copy_size, inp, copy_size);
          if (mem_ret != EOK) {
            KERNEL_LOG_ERROR(
                "Memcpy size[%zu] from inp[%llx] to out[%llx] failed.",
                copy_size, out, inp);
            ret = KERNEL_STATUS_INNER_ERROR;
            return;
          }
          out += size;
        }
        ++skipped_rows;
      }
      KERNEL_CHECK_FALSE_EXEC((out != out_end), return );
      if (out < out_start || out > out_end) {
        KERNEL_LOG_ERROR("Out[%llx] not in range[%llx, %llx)", out, out_start,
                         out_end);
        ret = KERNEL_STATUS_INNER_ERROR;
        return;
      }
      // Copy remaining data.
      std::vector<const T *> inp;
      inp.reserve(num_inputs);
      for (const auto &input : inputs) {
        inp.push_back(&(*input)(skipped_rows, 0));
      }
      const int64_t dim0 = output->dimension(0);
      for (int64_t i = skipped_rows; i < dim0; ++i) {
        for (int64_t j = 0; j < static_cast<int64_t>(num_inputs); ++j) {
          ptrdiff_t size = std::min(sizes[j], out_end - out);
          size_t copy_size = size * sizeof(T);
          auto mem_ret = memcpy_s(out, copy_size, inp[j], copy_size);
          if (mem_ret != EOK) {
            KERNEL_LOG_ERROR(
                "Memcpy size[%zu] from inp[%llx] to out[%llx] failed.",
                copy_size, inp[j], out);
            ret = KERNEL_STATUS_INNER_ERROR;
            return;
          }
          out += size;
          inp[j] += size;
          KERNEL_CHECK_FALSE_EXEC((out != out_end), return );
        }
      }

    };
    CpuKernelUtils::ParallelFor(ctx, output->size(), sizeof(T), work);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR,
                       "ConcatV2Kernel failed.");
    KERNEL_LOG_INFO("ConcatV2Kernel success.");
    return KERNEL_STATUS_OK;
  }

 private:
  DataType data_type_;
  int32_t input_dims_;
  int64_t n_;
  int64_t output_concat_dim_;
  int64_t axis_;
  int64_t inputs_flat_dim0_;
};
}  // namespace aicpu
#endif