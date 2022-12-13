#! /bin/bash

RUSTFLAGS="-Ctarget-cpu=skylake" cargo build --release;
