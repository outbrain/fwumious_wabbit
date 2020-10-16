#!/bin/bash
cargo build --release
python benchmark.py all all False
