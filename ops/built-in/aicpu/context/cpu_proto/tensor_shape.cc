/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of tensor shape
 */

#include "cpu_tensor_shape.h"
#include "tensor_shape_impl.h"

namespace aicpu {
TensorShape::TensorShape(TensorShapeImpl *impl) : impl_(impl) {}

/*
 * get dims value of tensor shape.
 */
std::vector<int64_t> TensorShape::GetDimSizes() const
{
    return impl_->GetDimSizes();
}

/*
 * set dims value to tensor shape.
 */
void TensorShape::SetDimSizes(const std::vector<int64_t> &dims)
{
    impl_->SetDimSizes(dims);
}

/*
 * get format value of tensor shape.
 */
Format TensorShape::GetFormat() const
{
    return impl_->GetFormat();
}

/*
 * set format value to tensor shape.
 */
void TensorShape::SetFormat(Format format)
{
    impl_->SetFormat(format);
}

/*
 * get unknown rank value of tensor shape.
 */
bool TensorShape::GetUnknownRank() const
{
    return impl_->GetUnknownRank();
}

/*
 * set unknown rank value to tensor shape.
 */
void TensorShape::SetUnknownRank(bool unknownRank)
{
    impl_->SetUnknownRank(unknownRank);
}

/*
 * get dims size of tensor shape.
 */
int32_t TensorShape::GetDims() const
{
    return impl_->GetDims();
}

/*
 * get dim value of tensor shape index dim.
 */
int64_t TensorShape::GetDimSize(int32_t index) const
{
    return impl_->GetDimSize(index);
}

/*
 * get data elements number.
 */
int64_t TensorShape::NumElements() const
{
    return impl_->NumElements();
}
}