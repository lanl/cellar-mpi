# MPI C++ API Design
## Introduction
This document will outline the design of the APIs in MPI C++. The goals of this
document are to:

1. Act as a reference so users can understand the opinions of MPI C++, and
   mentally map the API-space better.
2. Provide a foundational set of principles for how new APIs should be designed.
3. Enforce a rationale for different design decisions. Design decisions can be
   changed by proposing changes to this document.

## Packaging
### Header-Only
MPI C++ is a header-only library. The main arguments for this are:
1. Most of its benefits come from meta-template programming, which, in today's
   C++, can only be exported by-way of placing the implementation in a header
   file.
2. In general, the library intends to be a very thin-wrapper over MPI, providing
   additional usability, safety, and correctness guarantees that the user
   *should* already be writing themselves. This hopefully means the additional
   code-generation over what the user would have just written themsleves is
   minimal.
3. This makes it easier to integrate MPI C++ into any project, regardless of
   build system.

### Header Layout
We're targetting a layout where functionality for different MPI objects is
segmented into separate header files. We're doing this for a couple reasons.
First, it has a build-time benefit - this reduces the total amount of header
files that must be included when just a subset of functionality is used. Second,
it presents a good user-interface - if I want to use `mpi::Comm` functionality,
then I should `#include <mpi/comm.hpp>`.

MPI C++ reserves the right to include any MPI C++ header from any other MPI
C++ header. However, transitive includes will never be removed within a major
version change (or pre 1.0 minor version change). It also reserves the right to
include any STL header.

When including a public `mpi/foo.hpp` header file, a user can expect to use
*any* routine exposed by this header without having to include any other
headers.

### CMake
MPI C++ intends to integrate cleanly with existing CMake systems, and therefore
is "built" and tested using modern, target-based CMake. It also installs
CMake `<package>-config.cmake` files to make it easy to use `find_package` to
install it.

## Types
### C++ Objects
In principle, each `MPI_Foo` type maps to four different types in MPI C++.
- `mpi::Foo`
- `mpi::UniqueFoo`
- `mpi::SharedFoo` (with supporting `mpi::WeakFoo`)

`mpi::Foo` acts as a reference to an `MPI_Foo`, providing an object-oriented
C++-style interface to the `MPI_Foo` object. When the `mpi::Foo` object goes out
of scope, the `MPI_Foo` object is *not* deallocated. The user is responsible
for ensuring the correct ownership semantics are enforced.

`mpi::UniqueFoo` acts as the sole-owner of an `MPI_Foo`. It can be freely moved,
but not copied. When an `mpi::UniqueFoo` value is destroyed, it deallocates the
underlying `MPI_Foo` object. This is similar in behavior to `std::unique_ptr`.

`mpi::SharedFoo` allows multiple owners of an `MPI_Foo` using reference
counting. When an `mpi::SharedFoo` is copied, it adds an additional reference.
When an `mpi::SharedFoo` is destroyed, it decrements the reference count. When
the reference count reaches 0, the `MPI_Foo` object is deallocated.
`mpi::WeakFoo` objects can also be created to give safe references to the
`MPI_Foo` object to other components without giving them ownership. This is
similar to `std::shared_ptr` and `std::weak_ptr`.

An aside: The typical advice around whether to use "shared" or "unique" applies
here - you should always default to using `mpi::UniqueFoo` unless you have a
good reason to use `mpi::SharedFoo`. The reference counting for `mpi::SharedFoo`
has a small overhead, but more importantly it projects a lack of understanding
of the lifetimes of objects in your code.

Both `mpi::UniqueFoo` or `mpi::SharedFoo` can be converted to an
`mpi::Foo` explicitly by calling `deref()` on the object. The existing
`mpi::UniqueFoo` or `mpi::SharedFoo` retains ownership of the `MPI_Foo` handle
and it is up to the user to ensure that the "owning" object outlives the
"reference" object. This allows APIs that want to work with an `MPI_Foo` to do
so without having to be generic over any concrete owner type for an `MPI_Foo`.

All of the different object types have, more-or-less, the same member functions.
For example, all of the following calls to `barrier()` are allowed:
```c++
mpi::Comm reference = ...;
mpi::UniqueComm unique = ...;
mpi::SharedComm shared = ...;

reference.barrier();
unique.barrier();
shared.barrier();
```

All `MPI_Foo` C++ objects derive from `mpi::trait::Deref<Self, mpi::Foo>`.
This gives API developers two possible function signatures when they want to
accept an `mpi::Foo`, but don't want to take ownership of it:

The easiest version takes the `mpi::Foo` object directly:
```c++
void library_mpi_function(mpi::Comm comm) {
    // use comm
}

void caller() {
    mpi::UniqueComm my_comm = ...;
    library_mpi_function(my_comm.deref());
}
```

To make the interface slightly more ergonomic, you can also take a "generic"
Deref object. E.g.
```c++
template<typename From>
void library_mpi_function(mpi::trait::Deref<From, mpi::Comm> &comm_ref) {
    mpi::Comm comm = comm_ref.deref();
    // use comm
}

void caller() {
    mpi::UniqueComm my_comm = ...;
    library_mpi_function(my_comm);
}
```

If you'd like to provide the interface that takes a `Deref` object while still
getting the benefits of not defining the function in-line, you can split the
core of the routine out into a separate function that takes `mpi::Foo` object
directly, and have a small wrapper that calls `deref()` and forwards the result
to the function with the explicit signature.

### Mutability
MPI has four different ways to configure an implementation to handle threading:

| Thread Support        | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| MPI_THREAD_SINGLE     | Only one thread will execute                                                  |
| MPI_THREAD_FUNNELED   | Multiple threads, but only "main thread" calls MPI                            |
| MPI_THREAD_SERIALIZED | Process is multi-threaded, any thread may call MPI, but user serializes calls |
| MPI_THREAD_MULTIPLE   | Multiple threads calls MPI with no synchronization                            |

This is where tagging routines as `const` comes into play. Except when
running as `MPI_THREAD_MULTIPLE`, it's not safe to call any MPI routine from
multiple threads at the same time, and therefore, to be const-correct, basically
none of the MPI routines can be marked as const.

Currently, we take the position that all functions defined on an `mpi::Foo`
object that wrap an MPI routine are not marked as const - essentially all
`mpi::Foo` objects must be taken as mutable references. It's possible that we
could find a way to make these `const` for `MPI_THREAD_MULTIPLE`, but this
design space hasn't been explored yet.

## Names
### Namespaces
#### mpi
All functionality falls under the root `mpi` namespace. Any first-class object
in MPI falls directly under the `mpi` namespace. E.g. `MPI_Comm` is `mpi::Comm`
(and `mpi::UniqueComm` and `mpi::SharedComm`).

Capabilities that are only used by "advanced" MPI C++ code is further sorted
under additional sub-namespaces.

#### mpi::trait
The `mpi::trait` namespace contains types assisting in writing type-safe
generic code.