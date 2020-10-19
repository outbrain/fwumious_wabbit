#!/bin/bash
cargo build --release
python3 benchmark.py all all False
