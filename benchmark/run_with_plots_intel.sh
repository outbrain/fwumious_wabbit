#!/bin/bash
if [ $1 == "--rebuild" ]; then
  export RUSTFLAGS="-C opt-level=3 -C target-cpu=skylake"
  cargo build --release
fi

python3 benchmark.py fw all True
