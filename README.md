# MPI C++
MPI C++ is a modern C++ interface to MPI. It takes advantage of features like
templates, RAII, and inheritance to improve terseness and correctness of MPI
code written in C++.

## Example
```c++
void main() {
    mpi::init();

    mpi::Comm world = mpi::Comm::world();
    mpi::UniqueComm comm = world.dup();

    int const sum = comm.all_reduce(mpi::sum(), world.rank());

    ASSERT_EQ(comm.size() * (comm.size() - 1) / 2, sum);

    mpi::finalize();
}
```