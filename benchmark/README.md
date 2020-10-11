======================================== CPU Info ========================================
Physical cores: 4
Total cores: 8
Current Frequency: 2900.00Mhz
======================================== System Information ========================================
System: Darwin
Release: 19.6.0
Version: Darwin Kernel Version 19.6.0: Mon Aug 31 22:12:52 PDT 2020; root:xnu-6153.141.2~1/RELEASE_X86_64
Machine: x86_64
Processor: i386
vw train, no cache: 32.49 ± 0.14 seconds, 1098 ± 0.06 MB, 149.38 ± 0.13% CPU (5 runs)
fw train, no cache: 6.39 ± 0.12 seconds, 258 ± 0.02 MB, 99.96 ± 0.05% CPU (5 runs)
vw train, using cache: 25.15 ± 0.29 seconds, 1098 ± 0.02 MB, 164.98 ± 0.32% CPU (5 runs)
fw train, using cache: 2.23 ± 0.14 seconds, 258 ± 0.04 MB, 99.92 ± 0.16% CPU (5 runs)
vw predict, no cache: 60.38 ± 1.55 seconds, 138 ± 0.02 MB, 157.76 ± 0.29% CPU (5 runs)
fw predict, no cache: 5.33 ± 0.26 seconds, 257 ± 0.01 MB, 99.64 ± 0.19% CPU (5 runs)
vw predictions loss: 0.7063349973398133
fw predictions loss: 0.6944415718639689
![benchmark results](benchmark_results.png)
