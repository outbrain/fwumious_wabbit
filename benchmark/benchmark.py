import math
import os.path
import sys
from collections import defaultdict

import memit
import psutil
import platform
import generate
import gzip
import shutil


def rm_quietly(f):
    if os.path.isfile(f):
        os.remove(f)


def cleanup():
    rm_quietly("train.vw")
    rm_quietly("train.vw.gz")
    rm_quietly("train.vw.gz.cache")
    rm_quietly("train.vw.gz.fwcache")
    rm_quietly("easy.vw")
    rm_quietly("hard.vw")


def clean_caches():
    rm_quietly("train.vw.gz.cache")
    rm_quietly("train.vw.gz.fwcache")


def print_system_info():
    print("=" * 40, "CPU Info", "=" * 40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    # CPU usage

    print("=" * 40, "System Information", "=" * 40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")


def gzip_file(f):
    with open(f, 'rb') as f_in, gzip.open(f + ".gz", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def time_bash_cmd(cmd):
    return memit.memit(cmd)


def benchmark_one_train_cycle(vw_train_cmd, fw_train_cmd):
    clean_caches()
    res = dict()
    if benchmark_vw:
        res["vw"] = train_model(vw_train_cmd)
        # print(f"vw train time: {vw_train_time} seconds, using {vw_train_mem} KB and {vw_train_cpu} CPU")
        # print(f"vw train time - with cache: {vw_train_time_with_cache} seconds, using {vw_mem_with_cache} KB and {vw_cpu_with_cache} CPU")
    if benchmark_fw:
        res["fw"] = train_model(fw_train_cmd)
        # print(f"fw train time: {fw_train_time} seconds, using {fw_train_mem} KB and {fw_train_cpu} CPU")
        # print(f"fw train time - with cache: {fw_train_time_with_cache} seconds, using {fw_mem_with_cache} KB and {fw_cpu_with_cache} CPU")

    return res


def train_model(cmd):
    vw_results = dict()
    vw_results["no_cache"] = time_bash_cmd(cmd)
    vw_results["with_cache"] = time_bash_cmd(cmd)
    return vw_results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("syntax: benchmark fw|vw|all cleanup|generate|train|predict|all")
        exit()

    benchmark_fw = sys.argv[1] == "all" or sys.argv[1] == "fw"
    benchmark_vw = sys.argv[1] == "all" or sys.argv[1] == "vw"

    action = sys.argv[2]

    print_system_info()

    vw_train_cmd = "vw --data train.vw.gz -l 0.1 -b 25 -c --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor vw_model --save_resume --interactions AB"
    fw_train_cmd = "../target/release/fw --data train.vw.gz -l 0.1 -b 25 -c --adaptive --fastmath --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor fw_model --save_resume --interactions AB"

    if action == "cleanup" or action == "generate" or action == "all":
        cleanup()

    if action == "generate" or action == "all":
        print("generating dataset, this may take a while")
        generate.generate()
        gzip_file("train.vw")

    if action == "train" or action == "all":
        benchmark_means = defaultdict(dict)
        repetitions = 10
        train_results = []
        for _ in range(repetitions):
            train_result = benchmark_one_train_cycle(vw_train_cmd, fw_train_cmd)
            train_results.append(train_result)
            for model in train_result.keys():
                for setting in train_result[model].keys():
                    if setting not in benchmark_means[model].keys():
                        benchmark_means[model][setting] = [0., 0., 0.]

                    for i in range(len(benchmark_means[model][setting])):
                        benchmark_means[model][setting][i] += train_result[model][setting][i] / float(repetitions)

        benchmark_stds = defaultdict(dict)
        for train_result in train_results:
            for model in train_result.keys():
                for setting in train_result[model].keys():
                    if setting not in benchmark_stds[model].keys():
                        benchmark_stds[model][setting] = [0., 0., 0.]

                    for i in range(len(benchmark_stds[model][setting])):
                        benchmark_stds[model][setting][i] += math.pow(train_result[model][setting][i] - benchmark_means[model][setting][i], 2) / float(repetitions)

        for model in benchmark_stds.keys():
            for setting in benchmark_stds[model].keys():
                for i in range(len(benchmark_stds[model][setting])):
                    benchmark_stds[model][setting][i] = math.sqrt(benchmark_stds[model][setting][i])

        print(benchmark_means)
        print(benchmark_stds)

