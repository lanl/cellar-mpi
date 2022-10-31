#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

using namespace mpi;

TEST(RMA, GetPut) {
    auto comm = Comm::world();

    auto win = UniqueWin<rank_t>::allocate(comm, comm.size());
    std::fill(win.begin(), win.end(), comm.rank());

    win.lock_all_no_check();

    comm.barrier();

    std::vector<rank_t> ranks(comm.size());
    for (rank_t target = 0; target < comm.size(); target++) {
        win.get(&ranks[target], 1, target, target);
    }

    for (rank_t target = 0; target < comm.size(); target++) {
        EXPECT_EQ(target, ranks[target]);
    }

    auto const my_rank = comm.rank();
    for (rank_t target = 0; target < comm.size(); target++) {
        win.put(&my_rank, 1, target, my_rank);
    }

    comm.barrier();

    for (rank_t target = 0; target < comm.size(); target++) {
        EXPECT_EQ(target, win[target]);
    }

    win.unlock_all();
}

TEST(RMA, LockUnlock) {
    auto comm = Comm::world();
    auto const my_rank = comm.rank();

    // Dumb implementation of a reduce
    auto const local_win_size = comm.rank() == 0 ? 1 : 0;

    auto win = UniqueWin<rank_t>::allocate(comm, local_win_size);

    if (comm.rank() == 0) {
        win.lock(mpi::WinLockType::Exclusive,
                 0,
                 mpi::WinLockAssertFlags::NoCheck); // only rank 0 will be accessing
        rank_t const init = 0;
        win.put(&init, 1, 0, 0);
        win.unlock(0);
    }

    comm.barrier();

    win.lock(mpi::WinLockType::Exclusive, 0);
    {
        rank_t sum;
        win.get(&sum, 1, 0, 0);
        sum += 1;
        win.put(&sum, 1, 0, 0);
    }
    win.unlock(0);

    // wait for all increments to complete
    comm.barrier();

    rank_t sum;
    win.lock_all();
    win.get(&sum, 1, 0, 0);
    win.unlock_all();

    ASSERT_EQ(comm.size(), sum);
}