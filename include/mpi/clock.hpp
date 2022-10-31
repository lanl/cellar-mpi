/**
 * @file clock.hpp
 *
 * @brief Defines a C++ Clock using the MPI clock.
 * @date 2019-01-04
 *
 * @copyright Copyright (C) 2019 Triad National Security, LLC
 */

#ifndef MPI_CLOCK_HPP
#define MPI_CLOCK_HPP

#include "mpi_stub_out.h"

#include <chrono>

namespace mpi {

/**
 * @brief A clock satisfying the Clock and TrivialClock C++ concept, ala
 * std::chrono::high_resolution_clock, implemented using MPI_Wtime().
 */
class MpiClock {
  public:
    /**
     * @brief The MPI timer returns time as a double-precision floating point.
     */
    using rep = double;

    /**
     * @brief The MPI timer measures time in seconds, though the real tick duration is typically
     *  much shorter.
     */
    using period = std::chrono::seconds::period;

    /**
     * @brief MpiClock returns time as seconds represented as double-precision floating point.
     */
    using duration = std::chrono::duration<rep, period>;

    /**
     * @brief MpiClock time_point
     */
    using time_point = std::chrono::time_point<MpiClock>;

    /**
     * @brief MPI time is guaranteed to count upwards and at the same duration each tick.
     *
     * Call MpiClock::tick() to get the tick duration.
     */
    static const bool is_steady = true;

    /**
     * @brief Get the current time of the MPI timer in seconds.
     *
     * The time is not guaranteed to be "real" time - it's just guaranteed to count upwards.
     *
     * @return
     *  current time
     */
    static time_point now() { return time_point(duration(MPI_Wtime())); }

    /**
     * @brief Gets the resolution of the MPI timer in "seconds per tick".
     *
     * @return
     *  The resolution of the MPI timer in "seconds per tick".
     */
    static duration tick() { return duration(MPI_Wtick()); }
};
} // namespace mpi
#endif // MPI_CLOCK_HPP