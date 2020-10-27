#include "sparse_group.h"

namespace aicpu {
void GroupIterable::IteratorStep::UpdateEndOfGroup()
{
    ++next_loc_;
    const auto &ix_t = iter_->ix_matrix_;
    const int64_t N = ix_t.dimension(0);
    while (next_loc_ < N && iter_->GroupMatches(ix_t, loc_, next_loc_)) {
        ++next_loc_;
    }
}

bool GroupIterable::IteratorStep::operator != (const IteratorStep &rhs) const
{
    return (rhs.loc_ != loc_);
}

bool GroupIterable::IteratorStep::operator == (const IteratorStep &rhs) const
{
    return (rhs.loc_ == loc_);
}

GroupIterable::IteratorStep &GroupIterable::IteratorStep::operator ++ () // prefix ++
{
    loc_ = next_loc_;
    UpdateEndOfGroup();
    return *this;
}

const GroupIterable::IteratorStep GroupIterable::IteratorStep::operator ++ (int) // postfix ++
{
    IteratorStep lhs(*this);
    ++(*this);
    return lhs;
}

Group GroupIterable::IteratorStep::operator*() const
{
    return Group(iter_, loc_, next_loc_);
}

std::vector<int64_t> Group::group() const
{
    std::vector<int64_t> g;
    const auto &ix_t = iter_->ix_matrix_;
    for (const int d : iter_->group_dims_) {
        g.push_back(ix_t(loc_, d));
    }
    return g;
}

TTypes<int64_t>::UnalignedConstMatrix Group::indices() const
{
    return TTypes<int64_t>::UnalignedConstMatrix(&(iter_->ix_matrix_(loc_, 0)), next_loc_ - loc_, iter_->dims_);
}
} // namespace aicpu
