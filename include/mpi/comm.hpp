/**
 * @file comm.hpp
 *
 * @brief Defines types for using MPI_Comm.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_COMM_HPP
#define MPI_COMM_HPP

#include "mpi_stub_out.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
#include <optional>
#include <span>

#include "attrs.hpp"
#include "buffer.hpp"
#include "datatype.hpp"
#include "deref.hpp"
#include "exception.hpp"
#include "group.hpp"
#include "handle.hpp"
#include "keyval.hpp"
#include "op.hpp"
#include "request.hpp"
#include "status.hpp"

namespace mpi {
struct CommHandleTraits {
    using handle_t = MPI_Comm;

    static handle_t null() { return MPI_COMM_NULL; }
    static void destroy(handle_t &handle) { check_result(MPI_Comm_free(&handle)); }

    static bool is_system_handle(handle_t handle) {
        return (handle == MPI_COMM_WORLD || handle == MPI_COMM_SELF);
    }
};

struct CommAttrTraits {
    using handle_t = MPI_Comm;

    using copy_attr_function = MPI_Comm_copy_attr_function;
    using delete_attr_function = MPI_Comm_delete_attr_function;

    static copy_attr_function *null_copy_function() { return MPI_COMM_NULL_COPY_FN; }
    static delete_attr_function *null_delete_function() { return MPI_COMM_NULL_DELETE_FN; }

    static void create_keyval(copy_attr_function *copy_fn,
                              delete_attr_function *delete_fn,
                              key_t *keyval,
                              void *extra_state) {
        check_result(MPI_Comm_create_keyval(copy_fn, delete_fn, keyval, extra_state));
    }

    static void free_keyval(key_t *keyval) { check_result(MPI_Comm_free_keyval(keyval)); }

    static void set_attr(handle_t handle, key_t keyval, void *attribute_val) {
        check_result(MPI_Comm_set_attr(handle, keyval, attribute_val));
    }

    static void get_attr(handle_t handle, key_t keyval, void *attribute_val, int *flag) {
        check_result(MPI_Comm_get_attr(handle, keyval, attribute_val, flag));
    }

    static void delete_attr(handle_t handle, key_t keyval) {
        check_result(MPI_Comm_delete_attr(handle, keyval));
    }
};

class Comm;
class UniqueComm;

namespace internal {
/**
 * @brief Provides C++-style methods to MPI routines.
 *
 * @details
 * Do not use directly. Prefer `mpi::trait::Deref<T, mpi::Comm>` to be generic over all mpi::Comm
 * types.
 *
 * @tparam ConcreteType
 */
template <typename ConcreteType>
class CommImpl : public trait::Deref<ConcreteType, Comm>,
                 public internal::AttrsImpl<ConcreteType, CommAttrTraits> {
  public:
    /**
     * @brief Gets the rank of the local process.
     *
     * @return The rank of the process.
     */
    rank_t rank() const {
        int rank;
        check_result(MPI_Comm_rank(comm(), &rank));
        return rank;
    }

    /**
     * @brief Gets the number of processes in the communicator.
     *
     * @return The number of processes in the communicator.
     */
    rank_t size() const {
        int size;
        check_result(MPI_Comm_size(comm(), &size));
        return size;
    }

    /**
     * @brief Gets the raw MPI_Comm handle value for this communicator.
     *
     * @return MPI_Comm handle
     */
    MPI_Comm comm() const { return static_cast<ConcreteType const *>(this)->get_raw(); }

    /**
     * @brief Gets the group description of the MPI Communicator.
     *
     * @return A unique handle to the group description of the MPI Communicator.
     */
    UniqueGroup group() const {
        UniqueGroup g;
        check_result(MPI_Comm_group(comm(), g.addressof()));
        return g;
    }

    /**
     * @brief Duplicates the communicator using MPI_Comm_dup.
     *
     * @return A unique handle to the MPI Communicator.
     */
    UniqueComm dup();

    /**
     * @brief Creates a new MPI Communicator that is a subset of the current communicator using the
     * group.
     *
     * @tparam From The concrete type of the Group object (e.g. Group or UniqueGroup)
     * @param group An mpi::Group value
     * @return A new, separate mpi::UniqueComm
     */
    template <typename From>
    UniqueComm create(trait::Deref<From, Group> const &group);

    /**
     * @brief Aborts execution of all processes in the MPI communicator.
     *
     * @param errorcode An exit code for the MPI program.
     */
    [[noreturn]] void abort(int errorcode) {
        MPI_Abort(comm(), errorcode);
        std::exit(errorcode); // This won't actually be called, but we need it MPI_Abort itself
                              // hasn't been marked as noreturn.
    }

    /**
     * @brief Initiates a global, asynchronous barrier operation.
     *
     * @return UniqueRequest The request object tracking the asynchronous barrier.
     */
    UniqueRequest immediate_barrier() {
        UniqueRequest request;
        check_result(MPI_Ibarrier(comm(), request.addressof()));
        return request;
    }

    /**
     * @brief Synchronous, global barrier operation.
     */
    void barrier() { check_result(MPI_Barrier(comm())); }

    /**
     * @brief Retuns the upper-bound for tags that this communicator supports
     *
     * @return The maximum tag value, inclusive
     */
    tag_t tag_ub() {
        tag_t *ub;
        if (!this->get_attr(MPI_TAG_UB, ub)) {
            std::cerr << rank()
                      << ": Internal error: MPI did not provide an MPI_TAG_UB value as required"
                      << std::endl;
            abort(EXIT_FAILURE);
        }
        return *ub;
    }

    void gather(rank_t root, DynBuffer send, std::optional<DynBuffer> recv = std::nullopt) {
        if (root == rank()) {
            if (!recv) {
                std::cerr << rank()
                          << ": The root rank must supply a receive buffer to Comm::gather"
                          << std::endl;
                abort(EXIT_FAILURE);
            }

            if (recv && recv->size_int() < send.size_int() * this->size()) {
                std::cerr << rank() << ": recv (size = " << recv->size_int()
                          << ") is not large enough to receive " << this->size()
                          << " buffers of size " << send.size() << std::endl;
                abort(EXIT_FAILURE);
            }
        }

        check_result(
            MPI_Gather(send,
                       static_cast<int>(send.size_int()),
                       send.datatype(),
                       recv,
                       static_cast<int>(send.size_int()), // This is the size of any single buffer,
                                                          // not the size of the receive buffer
                       recv->datatype(),
                       root,
                       comm()));
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void gather(rank_t root,
                std::span<T const> send,
                std::optional<std::span<T>> recv = std::nullopt) {
        std::optional<DynBuffer> recv_buf =
            recv ? DynBuffer(MakeBuffer(*recv)) : std::nullopt;
        gather(root, DynBuffer(MakeBuffer(send)), recv_buf);
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void gather(rank_t root, T const &send) {
        gather(root, std::span<T const>(&send, 1));
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void gather_into_root(rank_t root, std::span<T const> send, std::span<T> recv) {
        if (root != rank()) {
            std::cerr
                << rank() << ": Comm::gather_into_root called from a rank other than the root ("
                << root
                << ") - gether_into_root should only be used on root rank, or you should call "
                   "gather with the receive buffer."
                << std::endl;
            abort(EXIT_FAILURE);
        }

        gather(root, send, recv);
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void gather_into_root(rank_t root, T const &send, std::span<T> recv) {
        gather_into_root(root, std::span<T const>(&send, 1), recv);
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    std::vector<T> gather_into_root(rank_t root, T const &send) {
        std::vector<T> recv(size());
        gather_into_root(root, send, recv);
        return recv;
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    std::vector<T> gather_into_root(rank_t root, std::span<T const> send) {
        std::vector<T> recv(send.size() * size());
        gather_into_root(root, send, recv);
        return recv;
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    std::vector<T> all_gather(T const &send) {
        std::vector<T> results(size());
        all_gather(send, std::span<T>(results));
        return results;
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void all_gather(T const &send, std::span<T> recv) {
        if (recv.size() < size()) {
            std::cerr << rank() << ": The recv buffer, of size " << recv.size()
                      << ", was smaller than the Comm::size() of " << size() << std::endl;
            abort(EXIT_FAILURE);
        }

        check_result(MPI_Allgather(&send,
                                   1,
                                   DatatypeTraits<T>::mpi_datatype(),
                                   recv.data(),
                                   1,
                                   DatatypeTraits<T>::mpi_datatype(),
                                   comm()));
    }

    void all_to_all(DynBuffer send, DynBuffer recv) {
        if (send.size() < size()) {
            std::cerr << rank() << ": The send buffer, of size " << send.size()
                      << ", must be at least of size " << size() << std::endl;
            abort(EXIT_FAILURE);
        }

        if (recv.size() < size()) {
            std::cerr << rank() << ": The recv buffer, of size " << recv.size()
                      << ", must be at least of size " << size() << std::endl;
            abort(EXIT_FAILURE);
        }

        check_result(
            MPI_Alltoall(send.data(), 1, send.datatype(), recv.data(), 1, recv.datatype(), comm()));
    }

    template <typename Send, typename Recv>
    void all_to_all(Send const &send, Recv &recv) {
        static_assert(are_buffers_compatible_v<Send, Recv>, "Buffer types are not compatible");

        all_to_all(MakeDynBuffer(send), MakeDynBuffer(recv));
    }

    template <typename Send>
    auto all_to_all(Send const &send)
        -> std::vector<std::remove_cv_t<typename decltype(MakeBuffer(send))::value_type>> {
        using T = std::remove_cv_t<typename decltype(MakeBuffer(send))::value_type>;
        std::vector<T> results(size());
        all_to_all(send, results);
        return results;
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void reduce(Op<OpTraits> const &op,
                rank_t root,
                T const send[],
                size_t send_count,
                T recv[] = nullptr,
                size_t recv_count = 0) {
        static_assert(OpTraits::template is_applicable<T>,
                      "The supplied data type is not valid for this MPI operation.");

        if (root == rank()) {
            if (!recv) {
                std::cerr << rank()
                          << ": The root rank must supply a receive buffer to Comm::gather"
                          << std::endl;
                abort(EXIT_FAILURE);
            }

            if (recv_count < send_count) {
                if (!recv) {
                    std::cerr << rank() << ": recv_count (" << recv_count
                              << ") is not large enough to receive " << size()
                              << " buffers of size " << send_count << std::endl;
                    abort(EXIT_FAILURE);
                }
            }
        }

        if (send_count > std::numeric_limits<int>::max()) {
            throw std::out_of_range("send array is too large");
        }

        check_result(MPI_Reduce(
            send, recv, send_count, DatatypeTraits<T>::mpi_datatype(), op.op(), root, comm()));
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void reduce(Op<OpTraits> const &op, rank_t root, T const &send) {
        reduce(op, root, &send, 1);
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void reduce(Op<OpTraits> const &op, rank_t root, std::vector<T> const &send) {
        reduce(op, root, send.data(), send.size());
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void reduce_into_root(Op<OpTraits> const &op,
                          rank_t root,
                          T const send[],
                          size_t send_count,
                          T recv[],
                          size_t recv_count) {
        if (root != rank()) {
            std::cerr
                << rank() << ": Comm::reduce_into_root called from a rank other than the root ("
                << root
                << ") - reduce_into_root should only be used on root rank, or you should call "
                   "reduce with the receive buffer."
                << std::endl;
            abort(EXIT_FAILURE);
        }

        reduce(op, root, send, send_count, recv, recv_count);
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    T reduce_into_root(Op<OpTraits> const &op, rank_t root, T send) {
        T recv;
        reduce_into_root(op, root, &send, 1, &recv, 1);
        return recv;
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    std::vector<T>
    reduce_into_root(Op<OpTraits> const &op, rank_t root, std::vector<T> const &send) {
        std::vector<T> recv(send.size());
        reduce_into_root(op, root, &send, 1, &recv, 1);
        return recv;
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void all_reduce(
        Op<OpTraits> const &op, T const send[], size_t send_count, T recv[], size_t recv_count) {
        static_assert(OpTraits::template is_applicable<T>,
                      "The supplied data type is not valid for this MPI operation.");

        if (send_count > std::numeric_limits<int>::max()) {
            throw std::out_of_range("send array is too large");
        }

        if (recv_count < send_count) {
            std::cerr << rank() << ": The recv buffer, of size " << recv_count
                      << ", must be at least of size " << send_count << std::endl;
            abort(EXIT_FAILURE);
        }

        check_result(MPI_Allreduce(
            send, recv, send_count, DatatypeTraits<T>::mpi_datatype(), op.op(), comm()));
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    T all_reduce(Op<OpTraits> const &op, T send) {
        T recv;
        all_reduce(op, &send, 1, &recv, 1);
        return recv;
    }

    template <typename OpTraits, typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    std::vector<T> all_reduce(Op<OpTraits> const &op, std::vector<T> const &send) {
        std::vector<T> recv(send.size());
        all_reduce(op, send.data(), send.size(), recv.data(), recv.size());
        return recv;
    }

    bool immediate_probe(rank_t source, tag_t tag, Status &status) {
        int flag;
        MPI_Status mpi_status;
        check_result(MPI_Iprobe(source, tag, comm(), &flag, &mpi_status));
        if (!flag) {
            return false;
        }

        status = Status(mpi_status);
        return true;
    }

    bool immediate_probe(rank_t source, Status &status) {
        return immediate_probe(source, 0, status);
    }

    bool immediate_probe_any(rank_t tag, Status &status) {
        return immediate_probe(MPI_ANY_SOURCE, tag, status);
    }

    bool immediate_probe_any(Status &status) { return immediate_probe_any(0, status); }

    template <typename T>
    UniqueRequest
    immediate_send(T const send[], std::size_t send_count, rank_t dest, tag_t tag = 0) {
        if (send_count > std::numeric_limits<int>::max()) {
            throw std::out_of_range("send array is too large");
        }

        UniqueRequest request;
        check_result(MPI_Isend(send,
                               static_cast<int>(send_count),
                               DatatypeTraits<T>::mpi_datatype(),
                               dest,
                               tag,
                               comm(),
                               request.addressof()));
        return request;
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    UniqueRequest immediate_send(T const &send, rank_t dest, tag_t tag = 0) {
        return immediate_send(&send, 1, dest, tag);
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    UniqueRequest immediate_recv(T recv[], std::size_t recv_count, rank_t source, tag_t tag = 0) {
        if (recv_count > std::numeric_limits<int>::max()) {
            throw std::out_of_range("receive array is too large");
        }

        UniqueRequest request;
        check_result(MPI_Irecv(recv,
                               recv_count,
                               DatatypeTraits<T>::mpi_datatype(),
                               source,
                               tag,
                               comm(),
                               request.addressof()));
        return request;
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    UniqueRequest immediate_recv(T &recv, rank_t source, tag_t tag = 0) {
        return immediate_recv(&recv, 1, source, tag);
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    Status recv_with_status(T recv[], std::size_t recv_count, rank_t source, tag_t tag = 0) {
        if (recv_count > std::numeric_limits<int>::max()) {
            throw std::out_of_range("receive array is too large");
        }

        MPI_Status status;
        check_result(MPI_Recv(
            recv, recv_count, DatatypeTraits<T>::mpi_datatype(), source, tag, comm(), &status));
        return Status(status);
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    Status recv_with_status(T &recv, rank_t source, tag_t tag = 0) {
        return recv_with_status(&recv, 1, source, tag);
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void recv(T recv[], std::size_t recv_count, rank_t source, tag_t tag = 0) {
        if (recv_count > std::numeric_limits<int>::max()) {
            throw std::out_of_range("receive array is too large");
        }

        check_result(MPI_Recv(recv,
                              recv_count,
                              DatatypeTraits<T>::mpi_datatype(),
                              source,
                              tag,
                              comm(),
                              MPI_STATUS_IGNORE));
    }

    template <typename T, typename = std::enable_if_t<is_datatype_v<T>>>
    void recv(T &recv, rank_t source, tag_t tag = 0) {
        recv(&recv, 1, source, tag);
    }

  protected:
    // Make it impossible to construct a CommImpl directly.
    CommImpl() = default;
};
} // namespace internal

/**
 * @brief Reference to an MPI Communicator object
 */
class Comm : public internal::Handle<CommHandleTraits>, public internal::CommImpl<Comm> {
    explicit Comm(MPI_Comm handle) : Handle(handle) {}

  public:
    Comm() = default;

    static Comm world() { return from_handle(MPI_COMM_WORLD); }
    static Comm from_handle(MPI_Comm handle) { return Comm{handle}; }
};

static_assert(sizeof(Comm) == sizeof(MPI_Comm), "Comm is expected to be the same size as MPI_Comm");

/**
 * @brief Owns a user MPI Communicator object
 */
class UniqueComm : public internal::UniqueHandle<CommHandleTraits>,
                   public internal::CommImpl<UniqueComm> {
    explicit UniqueComm(MPI_Comm handle) : UniqueHandle(handle) {}

  public:
    UniqueComm() = default;

    static UniqueComm from_handle(MPI_Comm handle) { return UniqueComm{handle}; }
};

static_assert(sizeof(UniqueComm) == sizeof(MPI_Comm),
              "UniqueComm is expected to be the same size as MPI_Comm");

template <typename ConcreteType>
UniqueComm internal::CommImpl<ConcreteType>::dup() {
    UniqueComm c;
    check_result(MPI_Comm_dup(comm(), c.addressof()));
    return c;
}

template <typename ConcreteType>
template <typename From>
UniqueComm internal::CommImpl<ConcreteType>::create(trait::Deref<From, Group> const &group) {
    UniqueComm c;
    check_result(MPI_Comm_create(comm(), group.deref(), c.addressof()));
    return c;
}
} // namespace mpi

#endif // MPI_COMM_HPP