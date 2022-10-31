/**
 * @file exception.hpp
 *
 * @brief Defines routines for propagating MPI errors as exceptions when MPI_ERRORS_RETURN is the
 *  MPI Errhandler.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_EXCEPTION_HPP
#define MPI_EXCEPTION_HPP

#include <exception>
#include <string>
#include <vector>

#include <mpi_stub_out.h>

namespace mpi {
/**
 * @brief Exception class for MPI Errors.
 */
class Exception : public std::exception {
  public:
    explicit Exception(int errorcode) : errorcode_(errorcode) {
        std::vector<char> err_string;
        err_string.resize(MPI_MAX_ERROR_STRING);

        int result_length;
        if (MPI_SUCCESS != MPI_Error_string(errorcode, err_string.data(), &result_length)) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        err_string.resize(result_length);
        err_string.push_back(0);
        what_ = std::move(err_string);
    }

    virtual const char *what() const noexcept override { return what_.data(); }

    int errorcode() const noexcept { return errorcode_; }

  private:
    int errorcode_;
    std::vector<char> what_;
};

inline void check_result(int errorcode) {
    if (MPI_SUCCESS != errorcode) {
        throw Exception(errorcode);
    }
}
} // namespace mpi

#endif // MPI_EXCEPTION_HPP