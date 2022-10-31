/**
 * @file op.hpp
 *
 * @brief Defines types for MPI_Op and routines for using built-in ops.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_OP_HPP
#define MPI_OP_HPP

#include "mpi_stub_out.h"
#include <type_traits>

#include "datatype.hpp"
#include "exception.hpp"
#include "handle.hpp"

namespace mpi {
struct OpHandleTraits {
    using handle_t = MPI_Op;

    static handle_t null() { return MPI_OP_NULL; }
    static void destroy(handle_t &handle) { check_result(MPI_Op_free(&handle)); }

    static bool is_system_handle(handle_t handle) {
        return (handle == MPI_MAX || handle == MPI_MIN || handle == MPI_SUM || handle == MPI_PROD ||
                handle == MPI_LAND || handle == MPI_BAND || handle == MPI_LOR ||
                handle == MPI_BOR || handle == MPI_LXOR || handle == MPI_BXOR ||
                handle == MPI_MAXLOC || handle == MPI_MINLOC);
    }
};

template <typename OpTraits>
class Op : public internal::UniqueHandle<OpHandleTraits> {
    explicit Op(MPI_Op handle) : UniqueHandle(handle) {}

  public:
    template <typename T>
    static constexpr bool is_applicable = OpTraits::template is_applicable<T>;

    Op() = default;
    Op(Op &&) = default;
    Op &operator=(Op &&) = default;

    ~Op() {
        if (!OpTraits::is_user_defined) into_raw();
    }

    static std::enable_if_t<!OpTraits::is_user_defined, Op> from_system_handle(MPI_Op op) {
        return Op{op};
    }

    MPI_Op op() const { return get_raw(); }
};

struct comparison_op_traits {
    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    static constexpr bool is_applicable =
        DatatypeTraits<T>::is_c_integer || DatatypeTraits<T>::is_floating_point;

    static constexpr bool is_user_defined = false;
};

using ComparisonOp = Op<comparison_op_traits>;

inline ComparisonOp max() { return ComparisonOp::from_system_handle(MPI_MAX); }
inline ComparisonOp min() { return ComparisonOp::from_system_handle(MPI_MIN); }

struct accumulate_op_traits {
    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    static constexpr bool is_applicable =
        DatatypeTraits<T>::is_c_integer || DatatypeTraits<T>::is_floating_point;

    static constexpr bool is_user_defined = false;
};

using AccumulateOp = Op<accumulate_op_traits>;

inline AccumulateOp sum() { return AccumulateOp::from_system_handle(MPI_SUM); }
inline AccumulateOp product() { return AccumulateOp::from_system_handle(MPI_PROD); }

struct logical_op_traits {
    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    static constexpr bool is_applicable =
        DatatypeTraits<T>::is_c_integer || DatatypeTraits<T>::is_logical;

    static constexpr bool is_user_defined = false;
};

using LogicalOp = Op<logical_op_traits>;

inline LogicalOp logical_and() { return LogicalOp::from_system_handle(MPI_LAND); }
inline LogicalOp logical_or() { return LogicalOp::from_system_handle(MPI_LOR); }
inline LogicalOp logical_xor() { return LogicalOp::from_system_handle(MPI_LXOR); }

struct bitwise_op_traits {
    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    static constexpr bool is_applicable = DatatypeTraits<T>::is_c_integer;

    static constexpr bool is_user_defined = false;
};

using BitwiseOp = Op<bitwise_op_traits>;

inline BitwiseOp bitwise_and() { return BitwiseOp::from_system_handle(MPI_BAND); }
inline BitwiseOp bitwise_or() { return BitwiseOp::from_system_handle(MPI_BOR); }
inline BitwiseOp bitwise_xor() { return BitwiseOp::from_system_handle(MPI_BXOR); }
} // namespace mpi

#endif // MPI_OP_HPP