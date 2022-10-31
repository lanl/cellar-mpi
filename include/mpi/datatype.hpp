/**
 * @file datatype.hpp
 *
 * @brief Defines trait object for MPI Datatypes for primitive C++ types.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_DATATYPE_HPP
#define MPI_DATATYPE_HPP

#include "mpi_stub_out.h"
#include <stdint.h>

#include <cstdint>
#include <type_traits>

namespace mpi {
using rank_t = int;
using tag_t = int;
using key_t = int;
using aint_t = MPI_Aint;

namespace internal {
struct DatatypeTraitsImpl {
    static constexpr bool is_datatype = true;
};
} // namespace internal

template <typename T, typename Enable = void>
struct DatatypeTraits {
    static constexpr bool is_datatype = false;
    static constexpr bool is_c_integer = false;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<bool> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_CXX_BOOL; }

    static constexpr bool is_c_integer = false;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = true;
};

template <>
struct DatatypeTraits<char> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_CHAR; }

    static constexpr bool is_c_integer = false;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<std::int8_t> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_INT8_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<std::int16_t> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_INT16_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<std::int32_t> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_INT32_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<std::int64_t> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_INT64_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<std::uint8_t> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_UINT8_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<std::uint16_t> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_UINT16_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

// This unfortunate monstronsity for uint32_t and uint64_t is required because with some compilers,
// size_t is just a typedef for the matching unsigned integer type, but on others it's a unique
// type. This enables size_t only when it does not match an existing type.
template <typename T>
struct DatatypeTraits<T,
                      std::enable_if_t<std::is_same<T, std::uint32_t>::value ||
                                       (std::is_same<T, std::size_t>::value &&
                                        sizeof(std::size_t) == sizeof(std::uint32_t))>>
    : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_UINT32_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <typename T>
struct DatatypeTraits<T,
                      std::enable_if_t<std::is_same<T, std::uint64_t>::value ||
                                       (std::is_same<T, std::size_t>::value &&
                                        sizeof(std::size_t) == sizeof(std::uint64_t))>>
    : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_UINT64_T; }

    static constexpr bool is_c_integer = true;
    static constexpr bool is_floating_point = false;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<float> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_FLOAT; }

    static constexpr bool is_c_integer = false;
    static constexpr bool is_floating_point = true;
    static constexpr bool is_logical = false;
};

template <>
struct DatatypeTraits<double> : public internal::DatatypeTraitsImpl {
    static MPI_Datatype mpi_datatype() { return MPI_DOUBLE; }

    static constexpr bool is_c_integer = false;
    static constexpr bool is_floating_point = true;
    static constexpr bool is_logical = false;
};

template <typename T>
static constexpr bool is_datatype_v = DatatypeTraits<T>::is_datatype;
} // namespace mpi

#endif // MPI_DATATYPE_HPP
