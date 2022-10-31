/**
 * @file attrs.hpp
 *
 * @brief Defines base class to support attributes on MPI objects that support it.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_ATTRS_HPP
#define MPI_ATTRS_HPP

#include "deref.hpp"
#include "keyval.hpp"

namespace mpi {
namespace internal {
template <typename ConcreteType, typename Traits>
class AttrsImpl {
    typename Traits::handle_t handle() const {
        return static_cast<ConcreteType const *>(this)->get_raw();
    }

    template <typename T>
    static std::enable_if_t<std::is_copy_constructible<T>::value,
                            typename Traits::copy_attr_function *>
    get_copy_fn() {
        return [](typename Traits::handle_t,
                  int,
                  void *,
                  void *attribute_val_in,
                  void *attribute_val_out,
                  int *flag) {
            auto const &val_in = *reinterpret_cast<T *>(attribute_val_in);
            *reinterpret_cast<T **>(attribute_val_out) = new T(val_in);
            *flag = 1;
            return MPI_SUCCESS;
        };
    }

    template <typename T>
    static std::enable_if_t<!std::is_copy_constructible<T>::value,
                            typename Traits::copy_attr_function *>
    get_copy_fn() {
        return Traits::null_copy_function();
    }

    template <typename T>
    static typename Traits::delete_attr_function *get_delete_fn() {
        return [](typename Traits::handle_t, int, void *attribute_val, void *) {
            delete reinterpret_cast<T *>(attribute_val);
            return MPI_SUCCESS;
        };
    }

  public:
    template <typename T>
    bool get_attr(key_t keyval, T &val) const {
        int flag = 0;
        void *result;
        Traits::get_attr(handle(), keyval, &result, &flag);

        if (flag) {
            val = *reinterpret_cast<T *>(&result);
            return true;
        }

        return false;
    }

    template <typename T, typename From>
    T *get_attr(trait::Deref<From, KeyVal<T>> const &keyval) const {
        T *value = nullptr;
        get_attr(keyval.deref(), value);
        return value;
    }

    template <typename T, typename From, typename... Args>
    T *create_attr(trait::Deref<From, KeyVal<T>> const &keyval, Args &&... arguments) {
        auto ptr = new T(std::forward<Args>(arguments)...);
        Traits::set_attr(handle(), keyval.deref(), ptr);
        return ptr;
    }

    template <typename T, typename From>
    T *set_attr(trait::Deref<From, KeyVal<T>> const &keyval, T &&arg) {
        return create_attr(keyval, std::forward<T>(arg));
    }

    template <typename T, typename From>
    void delete_attr(trait::Deref<From, KeyVal<T>> const &keyval) {
        Traits::delete_attr(handle(), keyval.deref());
    }

    static key_t create_keyval(typename Traits::copy_attr_function comm_copy_attr_fn,
                               typename Traits::delete_attr_function comm_delete_attr_fn,
                               void *context) {
        key_t keyval;
        Traits::create_keyval(comm_copy_attr_fn, comm_delete_attr_fn, &keyval, context);
        return keyval;
    }

    template <typename T>
    static std::enable_if_t<std::is_destructible<T>::value, UniqueKeyVal<T>> create_keyval() {
        return UniqueKeyVal<T>::from_handle(
            create_keyval(get_copy_fn<T>(), get_delete_fn<T>(), nullptr));
    }
};
} // namespace internal
} // namespace mpi

#endif // MPI_ATTRS_HPP