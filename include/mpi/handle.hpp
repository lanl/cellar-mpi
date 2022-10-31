/**
 * @file handle.hpp
 *
 * @brief Defines base types for the mpi classes that enhance the default MPI_ types.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_HANDLE_HPP
#define MPI_HANDLE_HPP

#include <cassert>
#include <utility>

namespace mpi {
namespace internal {
template <typename HandleTraits>
class Handle {
  public:
    using handle_traits = HandleTraits;
    using handle_t = typename HandleTraits::handle_t;

    void reset() { handle_ = HandleTraits::null(); }

    bool is_null() const { return handle_ == HandleTraits::null(); }
    operator bool() const { return !is_null(); }

    handle_t get_raw() const { return handle_; }

    handle_t *addressof() { return &handle_; }

    void free() {
        if (*this) {
            HandleTraits::destroy(handle_);
        }

        assert(is_null());
    }

    operator handle_t() const { return get_raw(); }

  protected:
    Handle() = default;
    Handle(handle_t handle_) : handle_(handle_) {}

    Handle(Handle const &other) = default;
    Handle &operator=(Handle const &other) = default;

    Handle(Handle &&other) = default;
    Handle &operator=(Handle &&other) = default;

  private:
    handle_t handle_ = HandleTraits::null();
};

template <typename HandleTraits>
class UniqueHandle {
  public:
    using handle_traits = HandleTraits;
    using handle_t = typename HandleTraits::handle_t;

  public:
    bool is_null() const { return handle_ == HandleTraits::null(); }
    operator bool() const { return !is_null(); }

    handle_t get_raw() const { return handle_; }

    handle_t into_raw() {
        auto handle = HandleTraits::null();
        std::swap(handle, this->handle_);
        return handle;
    }

    handle_t *addressof() { return &handle_; }

  protected:
    UniqueHandle() = default;
    UniqueHandle(handle_t handle) : handle_(handle) {}

    UniqueHandle(UniqueHandle const &other) = delete;
    UniqueHandle &operator=(UniqueHandle const &other) = delete;

    UniqueHandle(UniqueHandle &&other) { std::swap(handle_, other.handle_); }
    UniqueHandle &operator=(UniqueHandle &&other) {
        reset();
        std::swap(handle_, other.handle_);
        return *this;
    }

    ~UniqueHandle() { reset(); }

    void reset() {
        assert(!HandleTraits::is_system_handle(handle_));

        if (*this) {
            HandleTraits::destroy(handle_);
            assert(handle_ == HandleTraits::null());
        }

        assert(is_null());
    }

  private:
    handle_t handle_ = HandleTraits::null();
};
} // namespace internal
} // namespace mpi

#endif // MPI_HANDLE_HPP