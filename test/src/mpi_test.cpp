#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

using std::size_t;

using namespace mpi;

template <typename T>
void all_gather_test() {
    auto world = mpi::Comm::world();

    auto ranks = world.all_gather(static_cast<T>(world.rank()));

    for (size_t i = 0; i < ranks.size(); i++) {
        EXPECT_EQ(i, ranks[i]);
    }
}

TEST(AllGather, SupportsChar) { all_gather_test<char>(); }
TEST(AllGather, SupportsShort) { all_gather_test<short>(); }
TEST(AllGather, SupportsInt) { all_gather_test<int>(); }
TEST(AllGather, SupportsUint8) { all_gather_test<uint8_t>(); }
TEST(AllGather, SupportsUint16) { all_gather_test<uint16_t>(); }
TEST(AllGather, SupportsUint32) { all_gather_test<uint32_t>(); }
TEST(AllGather, SupportsUint64) { all_gather_test<uint64_t>(); }

template <typename T>
void all_to_all_test() {
    auto world = mpi::Comm::world();

    std::vector<T> const my_ranks(world.size(), static_cast<T>(world.rank()));
    auto ranks = world.all_to_all(my_ranks);

    for (size_t i = 0; i < ranks.size(); i++) {
        EXPECT_EQ(i, ranks[i]);
    }
}

TEST(AllToAll, SupportsChar) { all_to_all_test<char>(); }
TEST(AllToAll, SupportsShort) { all_to_all_test<short>(); }
TEST(AllToAll, SupportsInt) { all_to_all_test<int>(); }
TEST(AllToAll, SupportsUint8) { all_to_all_test<uint8_t>(); }
TEST(AllToAll, SupportsUint16) { all_to_all_test<uint16_t>(); }
TEST(AllToAll, SupportsUint32) { all_to_all_test<uint32_t>(); }
TEST(AllToAll, SupportsUint64) { all_to_all_test<uint64_t>(); }

TEST(Reduce, LogicalAnd) {
    auto world = mpi::Comm::world();

    if (world.rank() == 0) {
        EXPECT_FALSE(world.reduce_into_root(logical_and(), 0, false));
    } else {
        world.reduce(logical_and(), 0, true);
    }

    if (world.rank() == 0) {
        EXPECT_TRUE(world.reduce_into_root(logical_and(), 0, true));
    } else {
        world.reduce(logical_and(), 0, true);
    }
}

TEST(Reduce, Sum) {
    auto world = mpi::Comm::world();

    if (world.rank() == 0) {
        EXPECT_EQ(world.size() * (world.size() - 1) / 2,
                  world.reduce_into_root(sum(), 0, world.rank()));
    } else {
        world.reduce(sum(), 0, world.rank());
    }
}

TEST(AllReduce, LogicalOr) {
    auto world = mpi::Comm::world();

    EXPECT_FALSE(world.all_reduce(logical_or(), false));
    EXPECT_TRUE(world.all_reduce(logical_or(), world.rank() == 0));
    EXPECT_TRUE(world.all_reduce(logical_or(), true));

    // I would use std::vector here, but the darn std::vector<bool> specialization makes it
    // incompatible for working directly with MPI.
    bool my_false_true[2]{false, world.rank() == 0};
    bool false_true[2];
    world.all_reduce(logical_or(), my_false_true, 2, false_true, 2);

    EXPECT_FALSE(false_true[0]);
    EXPECT_TRUE(false_true[1]);
}

TEST(AllReduce, Sum) {
    auto world = mpi::Comm::world();

    EXPECT_EQ(world.size() * (world.size() - 1) / 2, world.all_reduce(sum(), world.rank()));
}

TEST(Immediate, Basic) {
    auto world = mpi::Comm::world();

    int neighbor;

    mpi::UniqueRequest send_request, recv_request;
    auto const rank = world.rank();
    if (world.rank() < world.size() - 1) {
        send_request = world.immediate_send(rank, world.rank() + 1);
    }

    if (world.rank() > 0) {
        recv_request = world.immediate_recv(neighbor, world.rank() - 1);
    }

    if (recv_request) {
        recv_request.wait();
        EXPECT_EQ(world.rank() - 1, neighbor);
    }

    if (send_request) send_request.wait();
}

TEST(Immediate, CollectiveWait) {
    auto world = mpi::Comm::world();

    auto const rank = world.rank();
    auto send_request = world.immediate_send(rank, 0);

    if (world.rank() == 0) {
        std::vector<int> ranks(world.size());

        std::vector<mpi::UniqueRequest> requests;
        requests.reserve(world.size());

        for (auto i = 0; i < world.size(); i++) {
            requests.push_back(world.immediate_recv(ranks[i], i));
        }

        auto statuses = mpi::wait_all_statuses(requests);
        for (auto &status : statuses) {
            EXPECT_TRUE(status.success());
        }

        for (auto i = 0; i < world.size(); i++) {
            EXPECT_EQ(i, ranks[i]);
        }
    }

    send_request.wait();
}

TEST(KeyVal, Rank) {
    auto rank_key_val = mpi::Comm::create_keyval<rank_t>();

    auto world = mpi::Comm::world();

    world.set_attr(rank_key_val, world.rank());
    EXPECT_TRUE(world.get_attr(rank_key_val));
    EXPECT_EQ(world.rank(), *world.get_attr(rank_key_val));

    auto duped = world.dup();

    EXPECT_TRUE(duped.get_attr(rank_key_val));

    EXPECT_EQ(world.rank(), *duped.get_attr(rank_key_val));
    EXPECT_EQ(world.rank(), *world.get_attr(rank_key_val));
}

TEST(KeyVal, NonCopyableStructure) {
    auto rank_key_val = mpi::Comm::create_keyval<std::unique_ptr<rank_t>>();

    auto world = mpi::Comm::world();

    world.set_attr(rank_key_val, std::make_unique<rank_t>(world.rank()));
    EXPECT_TRUE(world.get_attr(rank_key_val));
    EXPECT_TRUE(*world.get_attr(rank_key_val));
    EXPECT_EQ(world.rank(), **world.get_attr(rank_key_val));

    auto duped = world.dup();

    EXPECT_FALSE(duped.get_attr(rank_key_val));
}

TEST(KeyVal, TagUB) {
    auto world = mpi::Comm::world();

    EXPECT_LE(32767, world.tag_ub());
}