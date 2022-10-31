#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

TEST(Comm, Dup) {
    auto world = mpi::Comm::world();

    auto duped = world.dup();
}

TEST(Comm, RangeIncl) {
    auto world = mpi::Comm::world();

    if (auto front = world.create(world.group().range_incl(0, world.size() / 2))) {
        ASSERT_LE(world.rank(), world.size() / 2);

        ASSERT_EQ((front.size() - 1) * front.size() / 2,
                  front.all_reduce(mpi::sum(), world.rank()));
    } else {
        ASSERT_GT(world.rank(), world.size() / 2);
    }

    if (world.size() > 1) {
        if (auto back =
                world.create(world.group().range_incl(world.size() / 2 + 1, world.size() - 1))) {
            ASSERT_GE(world.rank(), world.size() / 2 + 1);

            auto expected_sum = (world.size() / 2 + world.size()) / 2 * back.size();
            ASSERT_EQ(expected_sum, back.all_reduce(mpi::sum(), world.rank()));
        } else {
            ASSERT_LT(world.rank(), world.size() / 2 + 1);
        }
    }
}