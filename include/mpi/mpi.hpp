/**
 * @file mpi.hpp
 *
 * @brief Umbrella header for whole MPI C++ library.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_MPI_HPP
#define MPI_MPI_HPP

#include "mpi_stub_out.h"

#include <chrono>
#include <cstdint>
#include <vector>
#include <optional>
#include <span>

#include "clock.hpp"
#include "comm.hpp"
#include "datatype.hpp"
#include "exception.hpp"
#include "group.hpp"
#include "op.hpp"
#include "request.hpp"
#include "status.hpp"
#include "win.hpp"

/**
 * @brief mpi is a library for writing massively parallel programs.
 */
namespace mpi {

/**
 * @brief Defines types that are not strictly MPI routines, but support modern C++ usage of the MPI
 * library.
 */
namespace trait {}

/**
 * @brief Initializes the MPI library. Must be called prior to calling any other `mpi::` routines.
 *
 * @param argc The program's argument count
 * @param argv The program's argument list
 *
 * @throws Exception
 */
inline void init(int &argc, char **&argv) { check_result(MPI_Init(&argc, &argv)); }

/**
 * @brief Finalizes the MPI library. Must be called prior to calling any other `mpi::` routines.
 *
 * @throws Exception
 */
inline void finalize() { check_result(MPI_Finalize()); }

/**
 * @brief Checks if MPI is initialized.
 *
 * @return True if MPI has been initialized, otherwise false.
 *
 * @throws Exception
 */
inline bool initialized() {
    int flag;
    check_result(MPI_Initialized(&flag));
    return flag != 0;
}

/**
 * @brief Checks if MPI is finalized.
 *
 * @return True if MPI has been finalized, otherwise false.
 *
 * @throws Exception
 */
inline bool finalized() {
    int flag;
    check_result(MPI_Finalized(&flag));
    return flag != 0;
}

// Multiple Completions

/**
 * @brief Waits for any request to complete, returning the index of that request and the status of
 *  the completed request in the `status` out parameter.
 *
 * @param requests A list of requests. At least one must be active.
 * @param status Returns the status of the completed request.
 * @return The index of the completed request.
 *
 * @throws Exception
 */
inline int wait_any(std::span<UniqueRequest> requests, Status &status) {
    if (requests.size() > std::numeric_limits<int>::max()) {
        throw std::out_of_range("requests array is too large");
    }

    int index;
    MPI_Status mpi_status;
    check_result(MPI_Waitany(static_cast<int>(requests.size()),
                             reinterpret_cast<MPI_Request *>(requests.data()),
                             &index,
                             &mpi_status));

    status = Status(mpi_status);

    return index;
}

/**
 * @brief Waits for any request to complete, returning the index of that request.
 *
 * @param requests A list of requests. At least one must be active.
 * @return The index of the completed request.
 *
 * @throws Exception
 */
inline int wait_any(std::span<UniqueRequest> requests) {
    if (requests.size() > std::numeric_limits<int>::max()) {
        throw std::out_of_range("requests array is too large");
    }

    int index;
    check_result(MPI_Waitany(static_cast<int>(requests.size()),
                             reinterpret_cast<MPI_Request *>(requests.data()),
                             &index,
                             MPI_STATUS_IGNORE));

    return index;
}

/**
 * @brief Waits for all requests to complete, returning in statuses the status of each completed
 *  request.
 *
 * @param requests A list of requests. At least one must be active.
 * @param statuses The status of each completed request. `statuses[i]` is the completion status for
 *  `requests[i]`.
 *
 * @throws Exception
 */
inline void wait_all(std::span<UniqueRequest> requests, std::span<Status> statuses) {
    if (requests.size() > std::numeric_limits<int>::max()) {
        throw std::out_of_range("requests array is too large");
    }

    if (statuses.size() < requests.size()) {
        throw std::logic_error(
            "statuses array must be large enough to hold a status for each request");
    }

    check_result(MPI_Waitall(static_cast<int>(requests.size()),
                             reinterpret_cast<MPI_Request *>(requests.data()),
                             reinterpret_cast<MPI_Status *>(statuses.data())));
}

/**
 * @brief Waits for all requests to complete.
 *
 * @param requests A list of requests. At least one must be active.
 *
 * @throws Exception
 */
inline void wait_all(std::span<UniqueRequest> requests) {
    if (requests.size() > std::numeric_limits<int>::max()) {
        throw std::out_of_range("requests array is too large");
    }

    check_result(MPI_Waitall(static_cast<int>(requests.size()),
                             reinterpret_cast<MPI_Request *>(requests.data()),
                             MPI_STATUSES_IGNORE));
}

/**
 * @brief Waits for all requests to complete, returning a list of each completed status.
 *
 * @param requests A list of requests. At least one must be active.
 * @return The status of each completed request. Element `i` is the completion status for
 *  `requests[i]`.
 *
 * @throws Exception
 */
inline std::vector<Status> wait_all_statuses(std::span<UniqueRequest> requests) {
    std::vector<mpi::Status> statuses(requests.size());
    wait_all(requests, statuses);
    return statuses;
}

/**
 * @brief Waits for at least one request to complete.
 *
 * @param requests A list of requests. At least one must be active.
 * @param indices The indices of each request that was completed.
 * @param statuses The status for each request that is completed, parallel to indices.
 * @return The number of completed requests.
 *
 * @throws Exception
 */
inline int wait_some(std::span<UniqueRequest> requests,
                     std::span<int> indices,
                     std::optional<std::span<Status>> statuses = std::nullopt) {
    if (requests.size() > std::numeric_limits<int>::max()) {
        throw std::out_of_range("requests array is too large");
    }

    if (indices.size() < requests.size()) {
        throw std::logic_error(
            "indices array must be large enough to hold an index for each request");
    }

    if (statuses) {
        if (statuses->size() < requests.size()) {
            throw std::logic_error(
                "statuses array must be large enough to hold a status for each request");
        }
    }

    MPI_Status *const statuses_data =
        statuses ? reinterpret_cast<MPI_Status *>(statuses->data()) : MPI_STATUSES_IGNORE;

    int num_completed;
    check_result(MPI_Waitsome(static_cast<int>(requests.size()),
                              reinterpret_cast<MPI_Request *>(requests.data()),
                              &num_completed,
                              indices.data(),
                              statuses_data));

    if (num_completed == MPI_UNDEFINED) {
        throw std::logic_error(
            "mpi::wait_some should only be called when there are still completed requests");
    }

    return num_completed;
}

/**
 * @brief Waits for at least one request to complete.
 *
 * @param requests A list of requests. At least one must be active.
 * @param indices A vector that receives the index of each completed request. It is not cleared -
 *  all new indices are opposed to the end of the list.
 *
 * @throws Exception
 */
inline void wait_some_into(std::span<UniqueRequest> requests, std::vector<int> &indices) {
    auto const original_indices_size = indices.size();
    // indices must be large enough to contain the completed indices for every Request in requests.
    // Even if some of the requests in the array have been completed, OpenMPI assumes that indices
    // is as large as requests.
    indices.resize(original_indices_size + requests.size());

    auto const num_completed =
        wait_some(requests, std::span<int>{indices}.subspan(original_indices_size));

    indices.resize(original_indices_size + num_completed);
}

/**
 * @brief Gets the current time of the MpiClock.
 *
 * @return Current time for MpiClock.
 */
inline std::chrono::time_point<MpiClock> wtime() { return MpiClock::now(); }

/**
 * @brief Gets the resolution of the MpiClock
 *
 * @return Resolution of MpiClock.
 */
inline std::chrono::duration<double> wtick() { return MpiClock::tick(); }
} // namespace mpi

#endif // MPI_MPI_HPP
