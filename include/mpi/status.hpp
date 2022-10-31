/**
 * @file status.hpp
 *
 * @brief Defines type for MPI_Status.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_STATUS_HPP
#define MPI_STATUS_HPP

#include "mpi_stub_out.h"

namespace mpi {

class Status {
    MPI_Status status;

  public:
    Status() {
        status.MPI_TAG = MPI_ANY_TAG;
        status.MPI_SOURCE = MPI_ANY_SOURCE;
        status.MPI_ERROR = MPI_SUCCESS;
    }

    explicit Status(MPI_Status status) : status(status) {}

    rank_t source() const { return status.MPI_SOURCE; }
    tag_t tag() const { return status.MPI_TAG; }
    int error() const { return status.MPI_ERROR; }

    bool success() const { return error() == MPI_SUCCESS; }
};

static_assert(sizeof(Status) == sizeof(MPI_Status),
              "mpi::Status must be convertible from an MPI_Status.");

inline Status *status_ignore() { return reinterpret_cast<Status *>(MPI_STATUS_IGNORE); }
inline Status *statuses_ignore() { return reinterpret_cast<Status *>(MPI_STATUSES_IGNORE); }

} // namespace mpi

#endif // MPI_STATUS_HPP