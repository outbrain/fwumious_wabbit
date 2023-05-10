#!/bin/bash
if [ $1 = "--rebuild" ]; then
  cargo build --release
fi
python3 benchmark.py fw all True
