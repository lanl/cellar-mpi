#!/bin/bash

git ls-files | grep -E "\.(cpp|hpp|h)$" | grep -v "gtest-mpi-listener.hpp" | xargs clang-format -i