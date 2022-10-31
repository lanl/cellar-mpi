/**
 * @file keyval.hpp
 *
 * @brief Defines a type for MPI keys.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_KEYVAL_HPP
#define MPI_KEYVAL_HPP

#include "datatype.hpp"
#include "deref.hpp"
#include "exception.hpp"
#include "handle.hpp"

namespace mpi {
struct KeyvalHandleTraits {
    using handle_t = key_t;

    static handle_t null() { return MPI_KEYVAL_INVALID; }
    static void destroy(handle_t &handle) { check_result(MPI_Comm_free_keyval(&handle)); }

    static bool is_system_handle(handle_t /*handle*/) { return false; }
};

template <typename T>
class KeyVal : public internal::Handle<KeyvalHandleTraits>, public trait::DerefSelf<KeyVal<T>> {
    KeyVal(key_t keyval) : Handle(keyval) {}

  public:
    using value_type = T;

    static KeyVal from_handle(key_t keyval) { return KeyVal(keyval); }
};

template <typename T>
class UniqueKeyVal : public internal::UniqueHandle<KeyvalHandleTraits>,
                     public trait::Deref<UniqueKeyVal<T>, KeyVal<T>> {
    UniqueKeyVal(key_t keyval) : UniqueHandle(keyval) {}

  public:
    using value_type = T;

    static UniqueKeyVal from_handle(key_t keyval) { return UniqueKeyVal(keyval); }
};
} // namespace mpi

#endif // MPI_KEYVAL_HPP
