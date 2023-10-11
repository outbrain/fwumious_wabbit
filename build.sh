#!/bin/bash

################################################################################################
# This file serves as an entrypoint for building the binary with specific rustc flags.		   #
# If there are flags you would like to test out, simply add them to RUSTFLAGS env. By default, #
# no flags are used (generic release build)													   #
################################################################################################

cargo build --release;

# Using specific flags examples
#Due to the addition of SIMD instructions and loop unrolling provides a 10 ~ 20 % speed up on machines that support avx512 instructions
#RUSTFLAGS="-O -C target-cpu=skylake-avx512 -C target-feature=+avx2,+avx,+fma" cargo build --release;

#RUSTFLAGS="-Ctarget-cpu=skylake" cargo build --release;
#RUSTFLAGS="-Ctarget-cpu=cascadelake" cargo build --release;
