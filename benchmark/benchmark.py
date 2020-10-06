import math
import os.path
import sys

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


def vw_clean_cache():
    rm_quietly("train.vw.gz.cache")


def fw_clean_cache():
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


def time_bash_cmd(cmd, proc_name):
    return memit.memit(cmd, proc_name)


def benchmark_cmd(cmd, proc_name, times, run_before=None):
    benchmark_means = [0., 0., 0.]
    results = []

    for _ in range(times):
        if run_before:
            run_before()

        result = time_bash_cmd(cmd, proc_name)
        results.append(result)

        for i in range(len(benchmark_means)):
            benchmark_means[i] += result[i] / float(times)

    benchmark_stds = [0., 0., 0.]

    for result in results:
        for i in range(len(benchmark_stds)):
            benchmark_stds[i] += math.pow(result[i] - benchmark_means[i], 2) / float(times)

    for i in range(len(benchmark_stds)):
        benchmark_stds[i] = math.sqrt(benchmark_stds[i])

    return benchmark_means, benchmark_stds


def format_metrics(times, means, stds):
    return f"{means[0]:.2f} ± {stds[0]:.2f} seconds, {means[1]/1024:.0f} ± {stds[1]/1024:.0f} MB, {means[2]:.2f} ± {stds[2]:.2f}% CPU ({times} runs)"


def cross_entropy(y_hat, y):
    return -math.log(y_hat) if y == 1 else -math.log(1 - y_hat)


def calc_loss(model_preds_file, input_file):
    model_preds = open(model_preds_file, 'rt')
    input = open(input_file, 'rt')

    loss = 0.
    i = 0
    for y_hat in model_preds:
        i += 1
        y = next(input).split("|")[0].strip()
        loss += cross_entropy(float(y_hat), float(y))

    return loss / float(i)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("syntax: benchmark fw|vw|all cleanup|generate|train|predict|train+predict|all")
        exit()

    benchmark_fw = sys.argv[1] == "all" or sys.argv[1] == "fw"
    benchmark_vw = sys.argv[1] == "all" or sys.argv[1] == "vw"

    action = sys.argv[2]

    print_system_info()

    vw_train_cmd = "vw --data train.vw.gz -l 0.1 -b 25 -c --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor vw_model --save_resume --interactions AB"
    fw_train_cmd = "../target/release/fw --data train.vw.gz -l 0.1 -b 25 -c --adaptive --fastmath --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor fw_model --save_resume --interactions AB"

    vw_predict_cmd = "vw --data easy.vw -t -p vw_preds.out --initial_regressor vw_model --hash all --interactions AB"
    fw_predict_cmd = "../target/release/fw --data easy.vw --sgd --adaptive -t -b 25 -p fw_preds.out --initial_regressor fw_model --link logistic --hash all --interactions AB"

    if action == "cleanup" or action == "generate" or action == "all":
        cleanup()

    if action == "generate" or action == "all":
        print("generating dataset, this may take a while")
        generate.generate()
        gzip_file("train.vw")

    times = 2

    if action == "train" or action == "train+predict" or action == "all":
        vw_train_no_cache_benchmark_means, vw_train_no_cache_benchmark_stds = benchmark_cmd(vw_train_cmd, "vw", times, vw_clean_cache)
        fw_train_no_cache_benchmark_means, fw_train_no_cache_benchmark_stds = benchmark_cmd(fw_train_cmd, "fw", times, fw_clean_cache)

        print(f"vw train, no cache: {format_metrics(times, vw_train_no_cache_benchmark_means, vw_train_no_cache_benchmark_stds)}")
        print(f"fw train, no cache: {format_metrics(times, fw_train_no_cache_benchmark_means, fw_train_no_cache_benchmark_stds)}")

        vw_train_with_cache_benchmark_means, vw_train_with_cache_benchmark_stds = benchmark_cmd(vw_train_cmd, "vw", times)
        fw_train_with_cache_benchmark_means, fw_train_with_cache_benchmark_stds = benchmark_cmd(fw_train_cmd, "fw", times)

        print(f"vw train, using cache: {format_metrics(times, vw_train_with_cache_benchmark_means, vw_train_with_cache_benchmark_stds)}")
        print(f"fw train, using cache: {format_metrics(times, fw_train_with_cache_benchmark_means, fw_train_with_cache_benchmark_stds)}")

    if action == "predict" or action == "train+predict" or action == "all":
        vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds = benchmark_cmd(vw_predict_cmd, "vw", times)
        fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds = benchmark_cmd(fw_predict_cmd, "fw", times)

        print(f"vw predict, no cache: {format_metrics(times, vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds)}")
        print(f"fw predict, no cache: {format_metrics(times, fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds)}")

        vw_model_loss = calc_loss("vw_preds.out", "easy.vw")
        print(f"vw predictions loss: {vw_model_loss}")

        fw_model_loss = calc_loss("fw_preds.out", "easy.vw")
        print(f"fw predictions loss: {fw_model_loss}")