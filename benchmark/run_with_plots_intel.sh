#!/bin/bash
export RUSTFLAGS="-C opt-level=3 -C target-cpu=skylake"
cargo build --release
python3 benchmark.py fw all True
