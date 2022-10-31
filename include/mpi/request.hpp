/**
 * @file request.hpp
 *
 * @brief Defines types for managing MPI_Request objects.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_REQUEST_HPP
#define MPI_REQUEST_HPP

#include "mpi_stub_out.h"

#include "datatype.hpp"
#include "exception.hpp"
#include "status.hpp"

#include <cstdlib>
#include <iostream>

namespace mpi {
struct RequestHandleTraits {
    using handle_t = MPI_Request;

    static handle_t null() { return MPI_REQUEST_NULL; }
    static void destroy(handle_t &handle) { check_result(MPI_Request_free(&handle)); }

    static bool is_system_handle(handle_t /*handle*/) { return false; }
};

class Request;
class UniqueRequest;

namespace internal {
template <typename ConcreteType>
class RequestImpl : public trait::Deref<ConcreteType, Request> {
    MPI_Request *addressof_internal() { return static_cast<ConcreteType *>(this)->addressof(); }

  public:
    MPI_Request request() const { return static_cast<ConcreteType const *>(this)->get_raw(); }

    void free() {
        if (request() != MPI_REQUEST_NULL) {
            check_result(MPI_Request_free(addressof_internal()));
        }
    }

    void wait() { check_result(MPI_Wait(addressof_internal(), MPI_STATUS_IGNORE)); }
    bool test() {
        int flag;
        check_result(MPI_Test(addressof_internal(), &flag, MPI_STATUS_IGNORE));
        return flag != 0;
    }

    Status wait_with_status() {
        MPI_Status status;
        check_result(MPI_Wait(addressof_internal(), &status));
        return Status(status);
    }

    bool test_with_status(Status &status) {
        MPI_Status mpistatus;
        int flag;
        check_result(MPI_Test(addressof_internal(), &flag, &mpistatus));

        if (flag != 0) {
            status = Status(mpistatus);
            return true;
        } else {
            return false;
        }
    }
};
} // namespace internal

class Request : public internal::Handle<RequestHandleTraits>,
                public internal::RequestImpl<Request> {
    explicit Request(MPI_Request request) : Handle(request) {}

  public:
    Request() = default;

    static Request from_handle(MPI_Request request) { return Request(request); }
};

static_assert(sizeof(Request) == sizeof(MPI_Request) && alignof(Request) == alignof(MPI_Request),
              "mpi::Request must be convertible to MPI_Request");

class UniqueRequest : public internal::UniqueHandle<RequestHandleTraits>,
                      public internal::RequestImpl<UniqueRequest> {
    explicit UniqueRequest(MPI_Request request) : UniqueHandle(request) {}

    void check_null() {
        if (!is_null()) {
            std::cout << "Requests must be completed before they're dropped!" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

  public:
    UniqueRequest() = default;

    UniqueRequest(UniqueRequest &&other) = default;

    UniqueRequest &operator=(UniqueRequest &&other) {
        check_null();
        UniqueHandle::operator=(std::move(other));
        return *this;
    }

    ~UniqueRequest() { check_null(); }

    static UniqueRequest from_handle(MPI_Request request) { return UniqueRequest(request); }
};

static_assert(sizeof(UniqueRequest) == sizeof(MPI_Request) &&
                  alignof(UniqueRequest) == alignof(MPI_Request),
              "mpi::UniqueRequest must be convertible to MPI_Request");
} // namespace mpi

#endif // MPI_REQUEST_HPP
