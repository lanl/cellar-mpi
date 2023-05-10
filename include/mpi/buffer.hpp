/**
 * @file buffer.hpp
 *
 * @brief Data buffers associated with data types.
 * @date 2019-01-18
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_BUFFER_HPP_
#define MPI_BUFFER_HPP_

// STL includes
#include <type_traits>
#include <vector>
#include <span>

// Internal includes
#include "datatype.hpp"

namespace mpi {
template <typename T>
class Buffer {
  public:
    using value_type = T;
    using mutable_value_type = std::remove_const_t<value_type>;

    explicit Buffer(std::span<T> data) : data_(data) {}

    MPI_Datatype datatype() const {
        return mpi::DatatypeTraits<mutable_value_type>::mpi_datatype();
    }

    constexpr T *data() { return data_.data(); }
    constexpr T const *data() const { return data_.data(); }

    constexpr size_t size() const { return data_.size(); }
    constexpr int size_int() const {
        if (size() > std::numeric_limits<int>::max()) {
            throw std::out_of_range("Buffer::size_int: Tried to send too large buffer.");
        }
        return static_cast<int>(size());
    }

  private:
    std::span<T> data_;
};

template <typename T>
Buffer<T> MakeBuffer(T &data) {
    return Buffer<T>(std::span<T>(&data, 1));
}

template <typename T>
Buffer<T const> MakeBuffer(T const &data) {
    return Buffer<T const>(std::span<T const>(&data, 1));
}

template <typename T>
Buffer<T> MakeBuffer(Buffer<T> data) {
    static_assert(is_datatype_v<std::remove_const_t<T>>,
                  "T does not implement mpi::DatatypeTraits");
    return data;
}

template <typename T>
Buffer<T> MakeBuffer(std::span<T> data) {
    static_assert(is_datatype_v<std::remove_const_t<T>>,
                  "T does not implement mpi::DatatypeTraits");
    return Buffer<T>(data);
}

template <typename T>
Buffer<T> MakeBuffer(std::vector<T> &data) {
    static_assert(is_datatype_v<std::remove_const_t<T>>,
                  "T does not implement mpi::DatatypeTraits");
    return Buffer<T>(std::span<T>(data));
}

template <typename T>
Buffer<T const> MakeBuffer(std::vector<T> const &data) {
    static_assert(is_datatype_v<std::remove_const_t<T>>,
                  "T does not implement mpi::DatatypeTraits");
    return Buffer<T const>(std::span<T const>(data));
}

class DynBuffer {
  public:
    template <typename Buf>
    DynBuffer(Buf const &buffer)
        : data_((void *)buffer.data()), count_(buffer.size_int()), datatype_(buffer.datatype()) {}

    DynBuffer(void *data, int count, MPI_Datatype datatype)
        : data_(data), count_(count), datatype_(datatype) {}

    MPI_Datatype datatype() { return datatype_; }

    void *data() { return data_; }
    void const *data() const { return data_; }

    size_t size() const { return count_; }
    int size_int() const { return count_; }

  private:
    void *data_;
    int count_;
    MPI_Datatype datatype_;
};

template <typename Buf>
DynBuffer MakeDynBuffer(Buf &buf) {
    return DynBuffer(MakeBuffer(buf));
}

template <typename Buf>
DynBuffer MakeDynBuffer(Buf const &buf) {
    return DynBuffer(MakeBuffer(buf));
}

template <typename Send, typename Recv>
struct are_buffers_compatible {
    static constexpr bool value =
        std::is_same<typename decltype(MakeBuffer(std::declval<Send>()))::mutable_value_type,
                     typename decltype(
                         MakeBuffer(std::declval<Recv>()))::mutable_value_type>::value;
};

template <typename Send>
struct are_buffers_compatible<Send, DynBuffer> {
    static constexpr bool value = true;
};

template <typename Recv>
struct are_buffers_compatible<DynBuffer, Recv> {
    static constexpr bool value = true;
};

template <>
struct are_buffers_compatible<DynBuffer, DynBuffer> {
    static constexpr bool value = true;
};

template <typename Send, typename Recv>
static constexpr bool are_buffers_compatible_v = are_buffers_compatible<Send, Recv>::value;
} // namespace mpi

#endif // MPI_BUFFER_HPP_