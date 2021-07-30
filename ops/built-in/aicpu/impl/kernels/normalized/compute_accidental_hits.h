/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_COMPUTE_ACCIDENTAL_HITS_H_
#define AICPU_KERNELS_NORMALIZED_COMPUTE_ACCIDENTAL_HITS_H_
#define FLT_MAX __FLT_MAX__

#include <utility>
#include "cpu_kernel.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
class ComputeAccidentalHitsMsCpuKernel : public CpuKernel {
 public:
  ~ComputeAccidentalHitsMsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;

  int64_t num_true_ = 0;
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> in_sampled_candidates_shape_;

  DataType x_dtype_ = DT_INT32;
  DataType weights_dtype_ = DT_INT32;

  template <typename T, typename S>
  static uint32_t CalComputeAccidentalHits(
      const CpuKernelContext &ctx, int64_t num_true,
      std::vector<Tensor *> &inputs,
      std::vector<Tensor *> &outputs,
      const std::vector<int64_t> &in_true_candidates_shape,
      const std::vector<int64_t> &in_sampled_candidates_shape) {
    const int64_t batch_size = in_true_candidates_shape[0];
    EigenTensor in_true_candidates(inputs[0], inputs[0]->GetData());
    EigenTensor in_sampled_candidates(inputs[1], inputs[1]->GetData());

    std::unordered_map<T, int> sampled_candidate_to_pos;
    for (int64_t i = 0; i < in_sampled_candidates_shape[0]; ++i) {
      sampled_candidate_to_pos[in_sampled_candidates.vec<T>()(i)] = i;
    }

    // Produce output in the same format as UnpackSparseFeatures.
    std::vector<T> indices;
    std::vector<T> ids;
    std::vector<S> weights;

    for (int64_t i = 0; i < batch_size; ++i) {
      for (int64_t j = 0; j < num_true; ++j) {
        const T true_candidate = in_true_candidates.matrix<T>()(i, j);
        const auto look = sampled_candidate_to_pos.find(true_candidate);
        if (look != sampled_candidate_to_pos.end()) {
          indices.push_back(i);
          ids.push_back(look->second);
          weights.push_back(static_cast<S>(-FLT_MAX));
        }
      }
    }

    EigenTensor out_indices(outputs[0], outputs[0]->GetData());
    EigenTensor out_ids(outputs[1], outputs[1]->GetData());
    EigenTensor out_weights(outputs[2], outputs[2]->GetData());

    for (size_t i = 0; i < indices.size(); ++i) {
      out_indices.vec<T>()(i) = indices[i];
      out_ids.vec<T>()(i) = ids[i];
      out_weights.vec<S>()(i) = weights[i];
    }

    Tensor *y0 = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(y0, KERNEL_STATUS_PARAM_INVALID,
                         "Get output:[0] failed.");
    auto y_shape = y0->GetTensorShape();
    KERNEL_CHECK_NULLPTR(y_shape, KERNEL_STATUS_PARAM_INVALID,
                         "Get output:[0] shape failed.");
    // judge unknown rank: y_shape->GetUnknownRank()
    y_shape->SetDimSizes({static_cast<int64_t>(indices.size())});

    Tensor *y1 = ctx.Output(1);
    KERNEL_CHECK_NULLPTR(y1, KERNEL_STATUS_PARAM_INVALID,
                         "Get output:[1] failed.");
    auto y_shape1 = y1->GetTensorShape();
    KERNEL_CHECK_NULLPTR(y_shape1, KERNEL_STATUS_PARAM_INVALID,
                         "Get output:[1] shape failed.");
    // judge unknown rank: y_shape1->GetUnknownRank()
    y_shape1->SetDimSizes({static_cast<int64_t>(ids.size())});

    Tensor *y2 = ctx.Output(2);
    KERNEL_CHECK_NULLPTR(y2, KERNEL_STATUS_PARAM_INVALID,
                         "Get output:[2] failed.");
    auto y_shape2 = y2->GetTensorShape();
    KERNEL_CHECK_NULLPTR(y_shape2, KERNEL_STATUS_PARAM_INVALID,
                         "Get output:[2] shape failed.");
    // judge unknown rank: y_shape2->GetUnknownRank()
    y_shape2->SetDimSizes({static_cast<int64_t>(weights.size())});

    return KERNEL_STATUS_OK;
  }
};  // ComputeAccidentalHitsMsCpuKernel

}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_COMPUTE_ACCIDENTAL_HITS_H_
