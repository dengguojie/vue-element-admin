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

/*!
 * \file strided_slice_infer_shape.h
 * \brief infer shape for strided_slice
 */

#ifndef OPS_BUILT_IN_OP_PROTO_STRIDED_SLICE_INFER_SHAPE_H_
#define OPS_BUILT_IN_OP_PROTO_STRIDED_SLICE_INFER_SHAPE_H_

#include <vector>
#include <string>
#include "../op_proto/util/op_common_util.h"


constexpr int32_t kStridedSliceNewAxis = -2;

// Sparse slicing specification
// if one does foo[3:5, ..., -3], this will have 3 length tensors
struct StridedSliceSparseSpec {
  int64_t dims;
  int32_t num_add_axis_after_ellipsis;
  const std::vector<int64_t> begin;
  const std::vector<int64_t> end;
  const std::vector<int64_t> strides;
  const uint64_t begin_mask;
  const uint64_t end_mask;
  uint64_t ellipsis_mask;
  const uint64_t new_axis_mask;
  const uint64_t shrink_axis_mask;
};

// Dense slicing specification
// all ellipses and new_axis' are expanded out. So if
// foo[3:5, ..., -3] where foo is 10 dimensional,
// each vector will have 10 entries whereas the
// sparse had 3 length tensors.
struct StridedSliceDenseSpec {
  const int64_t dims;
  uint64_t begin_mask;
  uint64_t end_mask;
  bool begin_valid;
  bool end_valid;
  std::vector<int64_t> begin;
  std::vector<int64_t> end;
  std::vector<int64_t> strides;

  // This vector helps construct the final shape of the slice.
  // The final tensor is reduced in rank whenever a single index e.g. foo[3]
  // is called for. The final tensor increases in rank with tf.newaxis
  // entries. If an index in this array is positive, the size of the dimension
  // is obtained from canonical end-begin. Otherwise, if it is a kNewAxis,
  // it will be 1. A shrunk dimension is skipped.
  std::vector<int64_t> final_shape_gather_indices;

  // The dense indexed shrink mask is which processing dimensions
  // should be shrunk. For example, if foo.shape = (10,10,10,10)
  // foo[3, ..., 5] has sparse_shrink_axis_mask of 0x5 and
  // dense_shrink_axis_mask of 0x9, yielding a final shape (10,10).
  uint64_t shrink_axis_mask;
};

struct StridedSliceParams {
  std::vector<int64_t> input_shape;
  std::vector<int64_t> begin;
  std::vector<int64_t> end;
  std::vector<int64_t> strides;
  std::vector<std::pair<int64_t, int64_t>> ranges;
  uint64_t begin_mask;
  uint64_t end_mask;
  uint64_t ellipsis_mask;
  uint64_t new_axis_mask;
  uint64_t shrink_axis_mask;
  bool begin_valid;
  bool end_valid;
  bool stride_valid;
  std::string to_string() const {
    std::string result = "input_shape:" + ops::to_string(input_shape);
    result += " begin:" + ops::to_string(begin);
    result += " end:" + ops::to_string(end);
    result += " strides:" + ops::to_string(strides);
    result += " ranges:" + ops::to_string(ranges);
    result += " begin_mask:" + std::to_string(begin_mask);
    result += " end_mask:" + std::to_string(end_mask);
    result += " ellipsis_mask:" + std::to_string(ellipsis_mask);
    result += " new_axis_mask:" + std::to_string(new_axis_mask);
    result += " shrink_axis_mask:" + std::to_string(shrink_axis_mask);
    result += " begin_valid:" + std::to_string(begin_valid);
    result += " end_valid:" + std::to_string(end_valid);
    result += " stride_valid:" + std::to_string(stride_valid);
    return result;
  }
};

static inline uint64_t bit1value(int i) {
  const uint64_t bit_i = static_cast<uint64_t>(1) << static_cast<uint64_t>(i);
  return bit_i;
}

static bool StridedSliceBuildDenseSpec(std::string op_name,
                                       const StridedSliceSparseSpec &sparse,
                                       StridedSliceDenseSpec *dense) {
  constexpr int32_t kShrinkAxis = -1;
  // Build expanded begin, end, strides, begin_mask, end_mask
  // to remove any ellipsis
  dense->begin.resize(dense->dims);
  dense->end.resize(dense->dims);
  dense->strides.resize(dense->dims);

  // What indices to get the final shape from.
  dense->begin_mask = 0;
  dense->end_mask = 0;
  dense->shrink_axis_mask = 0;

  int full_index = 0;
  for (int i = 0; i < sparse.dims; i++) {
    const uint64_t bit_i = bit1value(i);
    if (bit_i & sparse.ellipsis_mask) {
      // Expand the ellipsis into the appropriate indices
      // NOTE: this only works because we guaranteed one ellipsis
      int32_t next_index = std::min(dense->dims - (sparse.dims - i) + 1 +
                                        sparse.num_add_axis_after_ellipsis,
                                    dense->dims);
      for (; full_index < next_index; full_index++) {
        // new_axis' aren't real axis so you have to skip
        dense->begin[full_index] = dense->end[full_index] = 0;
        dense->strides[full_index] = 1;
        dense->begin_mask |= bit1value(full_index);
        dense->end_mask |= bit1value(full_index);
        dense->final_shape_gather_indices.push_back(full_index);
      }
    } else if (bit_i & sparse.new_axis_mask) {
      dense->final_shape_gather_indices.push_back(kStridedSliceNewAxis);
    } else {
      if (static_cast<size_t>(full_index) == dense->begin.size()) {
        OP_LOGE(op_name.c_str(), "Index out of range using input dim %d; input has only %lld dims.",
                full_index, dense->dims);
        return false;
      }

      // Gather slicing spec into appropriate index
      dense->begin[full_index] = sparse.begin[i];
      dense->end[full_index] = sparse.end[i];
      dense->strides[full_index] = sparse.strides[i];

      if (sparse.begin_mask & bit_i) {
        dense->begin_mask |= bit1value(full_index);
      }
      if (sparse.end_mask & bit_i) {
        dense->end_mask |= bit1value(full_index);
      }

      // If shrink, record where to get the dimensionality from (i.e.
      // new_axis creates a fake 1 size dimension. Also remember shrink
      // axis (now in dense form) so we can ignore dense->end below.
      if (sparse.shrink_axis_mask & bit_i) {
        dense->final_shape_gather_indices.push_back(kShrinkAxis);
        dense->shrink_axis_mask |= bit1value(full_index);
      } else {
        dense->final_shape_gather_indices.push_back(full_index);
      }
      full_index++;
    }
  }

  return true;
}

static std::pair<int64_t, int64_t> CalcRange(const std::vector<int64_t> &shape,
                                             const std::vector<int64_t> &begin,
                                             const std::vector<int64_t> &end,
                                             const std::vector<int64_t> &strides,
                                             const StridedSliceParams &params,
                                             const std::array<uint64_t, 2>& masks_i,
                                             int i) {
  const auto ranges = params.ranges;
  size_t index = static_cast<size_t>(i);
  if (index >= shape.size() || index >= ranges.size()) {
    return {0, -1};
  }

  if (shape[index] >= 0) {
    return {shape[index], shape[index]};
  }

  const auto shape_i = shape[index];
  auto begin_i = begin[index];
  auto end_i = end[index];
  auto stride_i = strides[index];
  const auto& range_i = ranges[index];
  auto get_output_interval = [shape_i, &begin_i, &end_i, &stride_i, masks_i](int64_t interval) -> int64_t {
    const std::array<int64_t, 2> valid_range = {
        {stride_i > 0 ? 0 : -1, stride_i > 0 ? interval : interval - 1}};
    auto canonical = [stride_i, interval, masks_i, valid_range](int64_t x, int c) ->int64_t {
      if (masks_i[c]) {
        return stride_i > 0 ? valid_range[c] : valid_range[static_cast<uint64_t>(c + 1) & static_cast<uint64_t>(1)];
      } else {
        int64_t x_fwd = x < 0 ? interval + x : x;  // make negative indices positive
        return x_fwd < valid_range[0]
               ? valid_range[0]
               : std::min(x_fwd, valid_range[1]);
      }
    };

    auto left = canonical(begin_i, 0);
    auto end = canonical(end_i, 1);
    auto output_interval = end - left;
    if ((output_interval < 0) != (stride_i < 0)) {
      return 0;
    }

    return output_interval / stride_i + (output_interval % stride_i != 0 ? 1 : 0);
  };

  int64_t range_left = get_output_interval(range_i.first);

  bool unknown_begin_i = !params.begin_valid && masks_i[0] == 0;
  if (unknown_begin_i) {
    range_left = 0;
    begin_i = 0;
  }

  bool unknown_end_i = !params.end_valid && masks_i[1] == 0;
  if (unknown_end_i) {
    range_left = 0;
    end_i = range_i.second;
  }

  if (!params.stride_valid) {
    range_left = 0;
    stride_i = 1;
  }

  int64_t range_right = -1;
  if (range_i.second != -1) {
    range_right = get_output_interval(range_i.second);
  } else {
    if (masks_i[0] != 0) {
      begin_i = 0;
    }

    if (masks_i[1] != 0) {
      end_i = -1;
    }

    if ((begin_i < 0) == (end_i < 0)) {
      range_right = get_output_interval(std::max(std::abs(end_i), std::abs(begin_i)));
    } else if (stride_i > 0) {
      if (begin_i < 0) {
        // stride_i > 0, range_i.second ==-1, begin_i < 0, end_i >= 0
        range_left = 0;
        if (begin_i + range_i.first >= end_i) {
          range_right = 0;
        } else {
          range_right = get_output_interval(std::min(std::abs(end_i), std::abs(begin_i)));
        }
      } else {
        // stride_i > 0, range_i.second ==-1, begin_i >= 0, end_i < 0
        range_right = -1;
      }
    } else if (stride_i < 0) {
      if (begin_i >= 0) {
        // stride_i < 0, range_i.second ==-1, begin_i >= 0, end_i < 0
        range_right = range_left;
        range_left = 1;
      } else {
        // stride_i < 0, range_i.second ==-1, begin_i < 0, end_i >= 0
        range_right = -1;
      }
    }
  }

  return {static_cast<int64_t>(std::min(static_cast<uint64_t>(range_left), static_cast<uint64_t>(range_right))),
          static_cast<int64_t>(std::max(static_cast<uint64_t>(range_left), static_cast<uint64_t>(range_right)))};
}

static bool StridedSliceCommonInferShape(std::string op_name,
                                         StridedSliceParams &params,
                                         std::vector<int64_t> &output_shape,
                                         std::vector<std::pair<int64_t, int64_t>> &output_ranges) {
  OP_LOGD(op_name.c_str(), "input params:%s.", params.to_string().c_str());
  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
  // i.e. there exists only no more than one ellipsis
  auto &ellipsis_mask = params.ellipsis_mask;
  if (ellipsis_mask && ((ellipsis_mask & (ellipsis_mask - 1)) != 0)) {
    OP_LOGE(op_name.c_str(), "Multiple ellipses in slice spec not allowed.");
    return false;
  }

  auto &begin = params.begin;
  auto &end = params.end;
  auto &strides = params.strides;
  auto &begin_mask = params.begin_mask;
  auto &end_mask = params.end_mask;
  auto &new_axis_mask = params.new_axis_mask;
  auto &shrink_axis_mask = params.shrink_axis_mask;
  auto &input_shape = params.input_shape;

  // Step 1: Account for ellipsis and new axis
  //
  // Check for ellipses and count how many non-newaxis' there are after
  bool ellipsis_seen = false;

  StridedSliceSparseSpec sparse_spec = {static_cast<int64_t>(strides.size()),
                                        0,
                                        begin,
                                        end,
                                        strides,
                                        begin_mask,
                                        end_mask,
                                        ellipsis_mask,
                                        new_axis_mask,
                                        shrink_axis_mask};

  for (int32_t i = 0; i < sparse_spec.dims; i++) {
    const uint64_t bit_i = bit1value(i);
    if (ellipsis_seen && (bit_i & new_axis_mask) != 0) {
      sparse_spec.num_add_axis_after_ellipsis++;
    }
    if (bit_i & ellipsis_mask) {
      ellipsis_seen = true;
    }
  }
  // If no ellipsis insert one at the end
  if (!ellipsis_seen) {
    sparse_spec.ellipsis_mask |= bit1value(sparse_spec.dims);
    sparse_spec.dims++;  // this effects loop iteration below
  }

  // Step 2: Make a sparse spec into a full index spec
  //
  // The sparse spec does not correspond to the number of dimensions
  // Make a dense spec that corresponds to the number of dimensions
  //
  // For example suppose foo[...,3:] on foo.shape=(2,2,3) then
  // we need to produce the missing begin_mask for the first two
  // dimensions i.e. from begin_mask_spec=0, end_mask_spec=2
  // we achieve begin_mask=6, end_mask=7
  StridedSliceDenseSpec dense_spec = {static_cast<int64_t>(input_shape.size()),
                                      0 /* begin_mask */,
                                      0 /* end_mask */,
                                      false /* begin_valid */,
                                      false /* end_valid */,
                                      begin,
                                      end,
                                      strides};

  // make sure begin and end always valid (has values)
  dense_spec.begin_valid = params.begin_valid;
  dense_spec.end_valid = params.end_valid;

  if (!StridedSliceBuildDenseSpec(op_name, sparse_spec, &dense_spec)) {
    return false;
  }
  begin = dense_spec.begin;
  end = dense_spec.end;
  strides = dense_spec.strides;

  // Step 3: Make implicit ranges (non-zero begin_masks and end_masks) explicit
  //         and bounds check!
  bool is_identity = true;
  bool slice_dim0 = true;
  bool is_simple_slice = true;
  vector<int64_t> processing_shape;
  vector<int64_t> processing_begin;
  vector<int64_t> processing_end;
  vector<int64_t> processing_strides;
  vector<std::pair<int64_t, int64_t>> processing_ranges;
  for (int i = 0; i < static_cast<int>(input_shape.size()); ++i) {
    auto &begin_i = begin[i];
    auto &end_i = end[i];
    auto &stride_i = strides[i];
    auto dim_i = input_shape[i];
    if (stride_i == 0) {
      OP_LOGE(op_name.c_str(), "strides[%d] must be non-zero", i);
      return false;
    }

    const uint64_t bit_i = bit1value(i);
    bool shrink_i = (dense_spec.shrink_axis_mask & bit_i);
    const std::array<uint64_t, 2> masks = {
        {dense_spec.begin_mask & bit_i, dense_spec.end_mask & bit_i}};
    if (dim_i == -1) {
      processing_shape.push_back(shrink_i ? 1 : -1);
      processing_begin.push_back(begin_i);
      processing_end.push_back(shrink_i ? (begin_i + 1) : end_i);
      processing_strides.push_back(shrink_i ? 1 : stride_i);
      if (shrink_i) {
        processing_ranges.push_back({1, 1});
      } else {
        auto unknown_range = CalcRange(processing_shape, processing_begin, processing_end, processing_strides,
                                       params, masks, i);
        processing_ranges.push_back(unknown_range);
      }
      continue;
    }

    const std::array<int64_t, 2> valid_range = {
        {stride_i > 0 ? 0 : -1, stride_i > 0 ? dim_i : dim_i - 1}};

    auto canonical = [stride_i, dim_i, masks, valid_range](int64_t x, int c) {
      if (masks[c]) {
        return stride_i > 0 ? valid_range[c] : valid_range[static_cast<uint64_t>(c + 1) & static_cast<uint64_t>(1)];
      } else {
        int64_t x_fwd = x < 0 ? dim_i + x : x;  // make negative indices positive
        return x_fwd < valid_range[0]
               ? valid_range[0]
               : std::min(x_fwd, valid_range[1]);
      }
    };

    if (shrink_i && stride_i <= 0) {
      OP_LOGE(op_name.c_str(), "only stride 1 allowed on non-range indexing.");
      return false;
    }
    is_simple_slice = is_simple_slice && (stride_i == 1);

    const bool begin_and_end_masked = (dense_spec.begin_mask & bit_i) && (dense_spec.end_mask & bit_i);
    if (dense_spec.begin_valid && dense_spec.end_valid) {
      if (shrink_i) {
        // If we are shrinking, the end index is now possibly incorrect. In
        // particular foo[-1] produces sparse_begin = -1, sparse_end = 0.
        // and canonical puts these to n-1 and 0, which implies a degenerate
        // interval. Fortunately, it is now safe to re-create end as begin+1.
        int64_t x_fwd = begin_i < 0 ? dim_i + begin_i : begin_i;
        begin_i = x_fwd;
        end_i = begin_i + 1;
        if (x_fwd < 0 || x_fwd >= dim_i) {
          OP_LOGE(op_name.c_str(), "slice index %lld of dimension %d  out of bounds.", begin_i, i);
          return false;
        }
      } else {
        begin_i = canonical(begin_i, 0);
        end_i = canonical(end_i, 1);
      }

      processing_begin.push_back(begin_i);
      processing_end.push_back(end_i);
      processing_strides.push_back(stride_i);

      // Update optimization values
      bool take_all_in_dimension =
          stride_i == 1 && begin_i == 0 && end_i == dim_i;
      is_identity = is_identity && take_all_in_dimension;
      slice_dim0 = slice_dim0 && ((i == 0 && stride_i == 1) || take_all_in_dimension);
    } else {
      is_identity = is_identity && (stride_i == 1 && begin_and_end_masked);
      slice_dim0 = slice_dim0 && ((i == 0 && stride_i == 1) || begin_and_end_masked);
      processing_begin.push_back(begin_i);
      processing_end.push_back(end_i);
      processing_strides.push_back(1);
    }

    // Compute the processing shape (the intermediate Eigen will produce)
    int64_t interval_length;
    bool known_interval = false;
    if (dense_spec.begin_valid && dense_spec.end_valid) {
      interval_length = end_i - begin_i;
      known_interval = true;
    } else if (shrink_i) {
      // The dimension is still known as 1 for the processing_shape, but will be
      // discarded for the final shape.
      interval_length = 1;
      known_interval = true;
    } else if (begin_and_end_masked) {
      // Even if we don't have values for begin or end, we do know that this
      // dimension covers the whole interval. If we have shape information for
      // this dimension, that tells us the interval length.
      if (dim_i >= 0) {
        if (stride_i < 0) {
          interval_length = -dim_i;
        } else {
          interval_length = dim_i;
        }
        known_interval = true;
      }
    }
    if (known_interval) {
      int64_t size_i;
      // Hold zero if the interval is degenerate, otherwise account for
      // remainder
      if (interval_length == 0 || ((interval_length < 0) != (stride_i < 0))) {
        size_i = 0;
      } else {
        size_i = interval_length / stride_i + (interval_length % stride_i != 0 ? 1 : 0);
      }
      processing_shape.push_back(size_i);
      processing_ranges.push_back({size_i, size_i});
    } else {
      processing_shape.push_back(-1);
      auto unknown_range = CalcRange(processing_shape, processing_begin, processing_end, processing_strides,
                                     params, masks, i);
      processing_ranges.push_back(unknown_range);
    }
  }

  // Step 4: Compute the final shape
  //
  // new_axis will increase dimension by 1 (with a one-size dimension)
  // slices like foo[3,...] will reduce dimension by 1.
  // This cannot be done earlier, because it depends on Step 3.
  vector<int64_t> final_output_shape;
  vector<std::pair<int64_t, int64_t>> final_output_ranges;
  vector<int64_t> final_input_shape;
  vector<int64_t> final_input_begin;
  vector<int64_t> final_input_end;
  vector<int64_t> final_input_strides;
  int shrink_gather_index = 0;
  for (auto gather_index : dense_spec.final_shape_gather_indices) {
    if (gather_index >= 0) {
      const auto dim_gather_i = processing_shape[gather_index];
      final_output_shape.push_back(dim_gather_i);
      final_output_ranges.push_back(dim_gather_i != -1 ?
                                    std::pair<int64_t, int64_t>(dim_gather_i, dim_gather_i)
                                                       : processing_ranges[gather_index]);

      final_input_shape.push_back(input_shape[gather_index]);
      final_input_begin.push_back(processing_begin[gather_index]);
      final_input_end.push_back(processing_end[gather_index]);
      final_input_strides.push_back(processing_strides[gather_index]);

      shrink_gather_index = gather_index + 1;
    } else if (gather_index == kStridedSliceNewAxis) {
      final_output_shape.push_back(1);
      final_output_ranges.push_back({1, 1});

      final_input_shape.push_back(1);
      final_input_begin.push_back(0);
      final_input_end.push_back(1);
      final_input_strides.push_back(1);
    } else {
      final_input_shape.push_back(input_shape[shrink_gather_index]);
      final_input_begin.push_back(processing_begin[shrink_gather_index]);
      final_input_end.push_back(processing_begin[shrink_gather_index] + 1);
      final_input_strides.push_back(1);

      shrink_gather_index += 1;
    }
  }

  output_shape = final_output_shape;
  output_ranges = final_output_ranges;
  input_shape = final_input_shape;
  begin = final_input_begin;
  end = final_input_end;
  strides = final_input_strides;


  return true;
}

#endif //OPS_BUILT_IN_OP_PROTO_STRIDED_SLICE_INFER_SHAPE_H_
