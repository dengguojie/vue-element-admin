/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of tensor shape impl
 */

#include "tensor_shape_impl.h"
#include "log.h"

namespace aicpu {
/*
 * get dims value of tensor shape.
 */
std::vector<int64_t> TensorShapeImpl::GetDimSizes() const
{
    std::vector<int64_t> ret;
    for (int32_t i = 0; i < tensorShape_->dim_size(); i++) {
        ret.emplace_back(tensorShape_->dim(i).size());
    }
    return ret;
}

/*
 * set dims value to tensor shape.
 */
void TensorShapeImpl::SetDimSizes(const std::vector<int64_t> &dims)
{
    tensorShape_->clear_dim();
    for (const auto &dim : dims) {
        aicpuops::TensorShape_Dim *aicpuDims = tensorShape_->add_dim();
        KERNEL_CHECK_NULLPTR_VOID(aicpuDims, "Protobuf add dim is null")
        aicpuDims->set_size(dim);
    }
}

/*
 * get format value of tensor shape.
 */
Format TensorShapeImpl::GetFormat() const
{
    return static_cast<Format>(tensorShape_->data_format());
}

/*
 * set format value to tensor shape.
 */
void TensorShapeImpl::SetFormat(Format format)
{
    tensorShape_->set_data_format(format);
}

/*
 * get unknown rank value of tensor shape.
 */
bool TensorShapeImpl::GetUnknownRank() const
{
    return tensorShape_->unknown_rank();
}

/*
 * set unknown rank value to tensor shape.
 */
void TensorShapeImpl::SetUnknownRank(bool unknownRank)
{
    tensorShape_->set_unknown_rank(unknownRank);
}

/*
 * get dims size of tensor shape.
 */
int32_t TensorShapeImpl::GetDims() const
{
    return tensorShape_->dim_size();
}

/*
 * get dim value of tensor shape index dim.
 */
int64_t TensorShapeImpl::GetDimSize(int32_t index) const
{
    if ((index >= GetDims()) || (index < 0)) {
        KERNEL_LOG_ERROR("dim index:%d must be not less than 0 and not greater than dims size:%d", index, GetDims());
        return 0;
    }

    return tensorShape_->dim(index).size();
}

/*
 * get data elements number.
 */
int64_t TensorShapeImpl::NumElements() const
{
    int64_t numElements = 1;
    for (int32_t i = 0; i < tensorShape_->dim_size(); i++) {
        int64_t dimSize = tensorShape_->dim(i).size();
        if (dimSize < 0) {
            return -1;
        }

        KERNEL_CHECK_ASSIGN_64S_MULTI(numElements, dimSize, numElements, -1);
    }
    return numElements;
}

/*
 * get tensor proto.
 * @return shared_ptr<TensorShapeProto>:tensor shape proto ptr
 */

aicpuops::TensorShape *TensorShapeImpl::GetProto() const
{
    return tensorShape_.get();
}
}