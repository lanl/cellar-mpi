/**
 * @file deref.hpp
 *
 *
 * @brief
 * Defines base classes to make it easy to take "Unique" handles that can be dereferenced to the
 * the generic, non-owning handle type.
 *
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_DEREF_HPP
#define MPI_DEREF_HPP

namespace mpi {
namespace trait {
template <typename From, typename To>
class Deref {
  public:
    To deref() { return To::from_handle(static_cast<From *>(this)->get_raw()); }
    To deref() const { return To::from_handle(static_cast<From const *>(this)->get_raw()); }
};

template <typename T>
class DerefSelf : public Deref<T, T> {};
} // namespace trait
} // namespace mpi

#endif // MPI_DEREF_HPP