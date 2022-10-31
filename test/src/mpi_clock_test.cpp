#include <gtest/gtest.h>
#include <mpi/mpi.hpp>

#include <thread>

using namespace std::literals::chrono_literals;

TEST(MpiClock, DeciSecond) {
    constexpr auto sleep_time = 100ms;

    auto const start = mpi::MpiClock::now();

    std::this_thread::sleep_until(start + sleep_time);

    auto const stop = mpi::MpiClock::now();

    // Yeah, yeah, this test isn't guaranteed to succeed, but I think that 50% is a plenty big
    // margin of error for sleep_until to make this pass all the time in practice.
    ASSERT_LE(sleep_time * 0.5, stop - start);
    ASSERT_GE(sleep_time * 1.5, stop - start);
}

TEST(MpiClock, ReasonableTick) {
    // We're expect to get a reasonable value for tick.
    auto const tick = mpi::MpiClock::tick();

    ASSERT_LT(0ms, tick);
    ASSERT_GE(1ms, tick);
}