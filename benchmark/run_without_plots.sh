#!/bin/bash
cargo build --release
python3 benchmark.py fw all False
