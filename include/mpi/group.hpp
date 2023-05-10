/**
 * @file group.hpp
 *
 * @brief Defines types and routines for building, querying, and using MPI Groups.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_GROUP_HPP
#define MPI_GROUP_HPP

#include "mpi_stub_out.h"

#include <cstddef>
#include <span>

#include "datatype.hpp"
#include "exception.hpp"
#include "handle.hpp"

namespace mpi {
class Group;
class UniqueGroup;

struct GroupRange {
    rank_t first = 0;
    rank_t last = -1;
    int stride = 1;

    GroupRange() = default;
    GroupRange(rank_t from, rank_t to, int stride) : first(from), last(to), stride(stride) {}

    static GroupRange unit_stride(rank_t from, rank_t to) {
        GroupRange r;
        r.first = from;
        r.last = to;
        return r;
    }
};

static_assert(sizeof(GroupRange) == sizeof(int[3]), "Range must be an int 3-tuple");
static_assert(offsetof(GroupRange, first) == 0, "first must be 0-offset");
static_assert(offsetof(GroupRange, last) == sizeof(int), "first must be 1-offset");
static_assert(offsetof(GroupRange, stride) == sizeof(int) * 2, "first must be 2-offset");

namespace internal {
template <typename ConcreteType>
class GroupImpl : public trait::Deref<ConcreteType, Group> {
  public:
    MPI_Group group() const { return static_cast<ConcreteType const *>(this)->get_raw(); }

    UniqueGroup range_incl(std::span<GroupRange> ranges) const;
    UniqueGroup range_incl(GroupRange &range) const;
    UniqueGroup range_incl(rank_t from, rank_t to) const;

    UniqueGroup range_excl(std::span<GroupRange> ranges) const;
    UniqueGroup range_excl(GroupRange &range) const;
    UniqueGroup range_excl(rank_t from, rank_t to) const;

    bool is_empty() const { return group() == MPI_GROUP_EMPTY; }
};
} // namespace internal

struct GroupHandleTraits {
    using handle_t = MPI_Group;

    static handle_t null() { return MPI_GROUP_NULL; }
    static void destroy(handle_t &handle) { check_result(MPI_Group_free(&handle)); }

    static bool is_system_handle(handle_t handle) { return handle == MPI_GROUP_EMPTY; }
};

class Group : public internal::Handle<GroupHandleTraits>, public internal::GroupImpl<Group> {
    explicit Group(MPI_Group group) : Handle(group) {}

  public:
    Group() = default;

    static Group empty() { return Group(MPI_GROUP_EMPTY); }

    static Group from_handle(MPI_Group group) { return Group(group); }
};

static_assert(sizeof(Group) == sizeof(MPI_Group),
              "Group is expected to be the same size as MPI_Group");

class UniqueGroup : public internal::UniqueHandle<GroupHandleTraits>,
                    public internal::GroupImpl<UniqueGroup> {
    explicit UniqueGroup(MPI_Group group) : UniqueHandle(group) {}

  public:
    UniqueGroup() = default;

    UniqueGroup(UniqueGroup &&) = default;
    UniqueGroup &operator=(UniqueGroup &&) = default;

    static UniqueGroup from_handle(MPI_Group group) { return UniqueGroup(group); }
};

static_assert(sizeof(UniqueGroup) == sizeof(MPI_Group),
              "UniqueGroup is expected to be the same size as MPI_Group");

template <typename ConcreteType>
UniqueGroup internal::GroupImpl<ConcreteType>::range_incl(std::span<GroupRange> ranges) const {
    if (ranges.size() > std::numeric_limits<int>::max()) {
        throw std::out_of_range("ranges array is too large");
    }

    UniqueGroup g;
    check_result(MPI_Group_range_incl(group(),
                                      static_cast<int>(ranges.size()),
                                      reinterpret_cast<int(*)[3]>(ranges.data()),
                                      g.addressof()));
    return g;
}

template <typename ConcreteType>
UniqueGroup internal::GroupImpl<ConcreteType>::range_incl(GroupRange &range) const {
    return range_incl(std::span<GroupRange>(&range, 1));
}

template <typename ConcreteType>
UniqueGroup internal::GroupImpl<ConcreteType>::range_incl(rank_t from, rank_t to) const {
    auto range = GroupRange::unit_stride(from, to);
    return range_incl(range);
}

template <typename ConcreteType>
UniqueGroup internal::GroupImpl<ConcreteType>::range_excl(std::span<GroupRange> ranges) const {
    if (ranges.size() > std::numeric_limits<int>::max()) {
        throw std::out_of_range("ranges array is too large");
    }

    UniqueGroup g;
    check_result(MPI_Group_range_excl(group(),
                                      static_cast<int>(ranges.size()),
                                      reinterpret_cast<int(*)[3]>(ranges.data()),
                                      g.addressof()));
    return g;
}

template <typename ConcreteType>
UniqueGroup internal::GroupImpl<ConcreteType>::range_excl(GroupRange &range) const {
    return range_excl(std::span<GroupRange>(&range, 1));
}

template <typename ConcreteType>
UniqueGroup internal::GroupImpl<ConcreteType>::range_excl(rank_t from, rank_t to) const {
    auto range = GroupRange::unit_stride(from, to);
    return range_excl(range);
}
} // namespace mpi

#endif // MPI_GROUP_HPP