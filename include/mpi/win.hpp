/**
 * @file win.hpp
 *
 * @brief Defines a type for allocating and managing MPI_Win types.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_WIN_HPP
#define MPI_WIN_HPP

#include "attrs.hpp"
#include "comm.hpp"
#include "exception.hpp"
#include "handle.hpp"

namespace mpi {
class Info {
    MPI_Info _info = MPI_INFO_NULL;

  public:
    MPI_Info info() const { return _info; }
};

enum class WinLockAssertFlags {
    None = 0,
    NoCheck = MPI_MODE_NOCHECK,
};

enum class WinLockType { Exclusive = MPI_LOCK_EXCLUSIVE, Shared = MPI_LOCK_SHARED };

template <typename T>
class Win;

template <typename T>
class UniqueWin;

struct WinHandleTraits {
    using handle_t = MPI_Win;

    static handle_t null() { return MPI_WIN_NULL; }
    static void destroy(handle_t &handle) { check_result(MPI_Win_free(&handle)); }

    static bool is_system_handle(handle_t /*handle*/) { return false; }
};

struct WinAttrTraits {
    using handle_t = MPI_Win;

    using copy_attr_function = MPI_Win_copy_attr_function;
    using delete_attr_function = MPI_Win_delete_attr_function;

    static copy_attr_function *null_copy_function() { return MPI_WIN_NULL_COPY_FN; }
    static delete_attr_function *null_delete_function() { return MPI_WIN_NULL_DELETE_FN; }

    static void create_keyval(copy_attr_function *copy_fn,
                              delete_attr_function *delete_fn,
                              key_t *keyval,
                              void *extra_state) {
        check_result(MPI_Win_create_keyval(copy_fn, delete_fn, keyval, extra_state));
    }

    static void free_keyval(key_t *keyval) { check_result(MPI_Win_free_keyval(keyval)); }

    static void set_attr(handle_t handle, key_t keyval, void *attribute_val) {
        check_result(MPI_Win_set_attr(handle, keyval, attribute_val));
    }

    static void get_attr(handle_t handle, key_t keyval, void *attribute_val, int *flag) {
        check_result(MPI_Win_get_attr(handle, keyval, attribute_val, flag));
    }

    static void delete_attr(handle_t handle, key_t keyval) {
        check_result(MPI_Win_delete_attr(handle, keyval));
    }
};

namespace internal {
template <typename ConcreteType, typename T>
class WinImpl : public trait::Deref<ConcreteType, Win<T>>,
                public internal::AttrsImpl<UniqueWin<T>, WinAttrTraits> {
  public:
    using value_type = T;
    using size_type = aint_t;
    using reference = T &;
    using const_reference = T const &;
    using pointer = T *;
    using const_pointer = T const *;
    using iterator = pointer;
    using const_iterator = const_pointer;

    MPI_Win win() const { return static_cast<ConcreteType const *>(this)->get_raw(); }

    T *base() {
        T *base;
        this->get_attr(MPI_WIN_BASE, base);
        return base;
    }

    T const *base() const {
        T const *base;
        this->get_attr(MPI_WIN_BASE, base);
        return base;
    }

    aint_t size() const {
        aint_t *size;
        this->get_attr(MPI_WIN_SIZE, size);
        return *size / sizeof(T);
    }

    /**
     * @brief Locks shared access to the `rank` portion of the window
     *
     * @param lock_type Exclusive or Shared
     * @param rank Portion of window to lock
     * @param flags Flags to pass to MPI_Win_lock_all
     */
    void
    lock(WinLockType lock_type, rank_t rank, WinLockAssertFlags flags = WinLockAssertFlags::None) {
        check_result(
            MPI_Win_lock(static_cast<int>(lock_type), rank, static_cast<int>(flags), win()));
    }

    /**
     * @brief Locks shared acccess to whole window
     *
     * @param flags Flags to pass to MPI_Win_lock_all
     */
    void lock_all(WinLockAssertFlags flags = WinLockAssertFlags::None) {
        check_result(MPI_Win_lock_all(static_cast<int>(flags), win()));
    }

    /**
     * @deprecated since 0.3.1
     */
    void lock_all_no_check() { lock_all(WinLockAssertFlags::NoCheck); }

    void unlock_all() { check_result(MPI_Win_unlock_all(win())); }

    void unlock(rank_t rank) { check_result(MPI_Win_unlock(rank, win())); }

    void flush_all() { check_result(MPI_Win_flush_all(win())); }

    void get(T recv[], std::size_t recv_count, rank_t target, aint_t target_disp) {
        check_result(MPI_Get(recv,
                             recv_count,
                             DatatypeTraits<T>::mpi_datatype(),
                             target,
                             target_disp,
                             recv_count,
                             DatatypeTraits<T>::mpi_datatype(),
                             win()));
    }

    void put(T const send[], std::size_t send_count, rank_t target, aint_t target_disp) {
        check_result(MPI_Put(send,
                             send_count,
                             DatatypeTraits<T>::mpi_datatype(),
                             target,
                             target_disp,
                             send_count,
                             DatatypeTraits<T>::mpi_datatype(),
                             win()));
    }

    reference operator[](size_type i) { return base()[i]; }
    const_reference operator[](size_type i) const { return base()[i]; }

    iterator begin() { return base(); }
    iterator end() { return base() + size(); }

    const_iterator cbegin() const { return base(); }
    const_iterator cend() const { return base() + size(); }
};
} // namespace internal

template <typename T>
class Win : public internal::Handle<WinHandleTraits>, public internal::WinImpl<Win<T>, T> {
    explicit Win(MPI_Win win) : Handle(win) {}

  public:
    static Win from_handle(MPI_Win win) { return Win(win); }
};

template <typename T>
class UniqueWin : public internal::UniqueHandle<WinHandleTraits>,
                  public internal::WinImpl<UniqueWin<T>, T> {
    explicit UniqueWin(MPI_Win win) : UniqueHandle(win) {}

  public:
    template <typename From>
    static UniqueWin
    allocate(trait::Deref<From, Comm> &comm, aint_t count, Info const &info = Info{}) {
        T *baseptr;
        MPI_Win win;
        check_result(MPI_Win_allocate(count * sizeof(T),
                                      sizeof(T),
                                      info.info(),
                                      comm.deref().comm(),
                                      (void **)&baseptr,
                                      &win));

        return UniqueWin{win};
    }

    static UniqueWin from_handle(MPI_Win win) { return UniqueWin(win); }
};
} // namespace mpi

#endif // MPI_WIN_HPP
