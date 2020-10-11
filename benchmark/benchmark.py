import math
import os.path
import sys

from measure import measure
import psutil
import platform
import generate
import gzip
import shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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
    print("### CPU Info")
    print("```")
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    print("```")

    print("### System Information")
    uname = platform.uname()
    print("```")
    print(f"System: {uname.system}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    print("```")


def gzip_file(f):
    with open(f, 'rb') as f_in, gzip.open(f + ".gz", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def time_bash_cmd(cmd, proc_name):
    return measure(cmd, proc_name)


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


def format_metricsformat_metrics(times, means, stds):
    return f"{means[0]:.2f} ± {stds[0]:.2f} seconds, {means[1]/1024.:.0f} ± {stds[1]/1024.:.2f} MB, {means[2]:.2f} ± {stds[2]:.2f}% CPU ({times} runs)"


def format_metrics_row(action, means, stds):
    return f"{action}|{means[0]:.2f} ± {stds[0]:.2f} | {means[1]/1024.:.0f} ± {stds[1]/1024.:.2f} | {means[2]:.2f} ± {stds[2]:.2f}"


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


def plot_results(vw_mem_values, fw_mem_values, vw_cpu_values, fw_cpu_values):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    import numpy as np
    ind = np.arange(max(len(fw_time_values), len(vw_time_values)))
    width = 0.35

    if benchmark_vw:
        ax1.bar(ind, vw_time_values, width, label='Vowpal Wabbit')
        ax2.bar(ind, vw_mem_values, width, label='Vowpal Wabbit')
        ax3.bar(ind, vw_cpu_values, width, label='Vowpal Wabbit')
    if benchmark_fw:
        ax1.bar(ind + width, fw_time_values, width, label='Fwumious Wabbit')
        ax2.bar(ind + width, fw_mem_values, width, label='Fwumious Wabbit')
        ax3.bar(ind + width, fw_cpu_values, width, label='Fwumious Wabbit')
    ax1.set_ylabel('seconds')
    ax2.set_ylabel('MB')
    ax3.set_ylabel('CPU %')
    ax1.set_title('Total runtime')
    ax2.set_title('Max memory use')
    ax3.set_title('Max CPU utilization')
    ax1.set_xticks(ind + width / 2)
    ax1.set_xticklabels(actions)
    ax2.set_xticks(ind + width / 2)
    ax2.set_xticklabels(actions)
    ax3.set_xticks(ind + width / 2)
    ax3.set_xticklabels(actions)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    plt.tight_layout(w_pad=3.0)
    plt.savefig('benchmark_results.png')
    return "benchmark_results.png"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("syntax: benchmark fw|vw|all cleanup|generate|train|predict|train+predict|all")
        exit()

    benchmark_fw = sys.argv[1] == "all" or sys.argv[1] == "fw"
    benchmark_vw = sys.argv[1] == "all" or sys.argv[1] == "vw"

    action = sys.argv[2]

    with open("README.md", "w") as readme:
        readme.write("")  # clear file
        sys.stdout = readme
        print("## Setup\n")

        print_system_info()

        vw_train_cmd = "vw --data train.vw.gz -l 0.1 -p blah.out -b 25 -c --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor vw_model --save_resume --interactions AB"
        fw_train_cmd = "../target/release/fw --data train.vw.gz -l 0.1 -p blah.out -b 25 -c --adaptive --fastmath --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor fw_model --save_resume --interactions AB"

        vw_predict_cmd = "vw --data easy.vw -t -p vw_preds.out --initial_regressor vw_model --hash all --interactions AB"
        fw_predict_cmd = "../target/release/fw --data easy.vw --sgd --adaptive -t -b 25 -p fw_preds.out --initial_regressor fw_model --link logistic --hash all --interactions AB"

        if action == "cleanup" or action == "generate" or action == "all":
            cleanup()

        if action == "generate" or action == "all":
            train_examples = 10000000
            test_examples = 10000000

            num_animals = 100000
            num_foods = 100000

            print("### Dataset details")
            print(f"we generate a synthetic dataset with {train_examples} train records and {test_examples} test records.\n")
            print("the task is 'Eat-Rate prediction' - each record describes the observed result of a single feeding experiment.\n")
            print("each record is made of a type of animal, a type of food, and a label indicating whether the animal ate the food.\n")
            print("the underlying model is simple - animals are either herbivores or carnivores - regardless of specific animal name,")
            print("and food is either plant based or meat based regardless of it's identity.")
            print("herbivores always eat plants (and only plants), and carnivores always eat meat (and only meat).\n")
            print("for convenience reasons, we name the animals 'Herbivore-1234' and 'Carnivore-5678', and the food items 'Plant-678'")
            print(" and 'Meat-234' so the expected outcome for a record is always clear.\n")
            print(f"there are {num_animals} animal types, and {num_foods} food types.")
            print("\n")

            generate.generate(train_examples, test_examples)
            with open("train.vw", "r") as dataset:
                print("see for example the first 5 lines from the train dataset (after some pretty-printing):")
                print("label|animal|food")
                print("-----|------|----")
                for _ in range(5):
                    print(next(dataset).strip("\n"))
                print("\n")
            gzip_file("train.vw")

        times = 10
        actions = []
        vw_time_values = []
        vw_mem_values = []
        vw_cpu_values = []
        fw_time_values = []
        fw_mem_values = []
        fw_cpu_values = []

        print("## Results\n")
        if action == "train" or action == "train+predict" or action == "all":
            print("we measure first 3 scenarios:")
            print("1. train a new model from a gzipped dataset, generating a gzipped cache file for future runs, and an output model file - *this is a typical scenario in our AutoML system - we start by generating the cache file for the next runs.*")
            print("1. train a new model over the dataset in the gzipped cache, and generate an output model - *this is also a typical scenario - we usually run many concurrent model evaluations as part of the model search*")
            print("1. use a generated model to make predictions over a dataset read from a text file, and print them to an output predictions file - *this is to illustrate potential serving performance, we don't usually predict from file input as our offline flows always apply online learning. note that when running as daemon we use half as much memory since training gradients are not loaded - only model weights*")
            print("\n")

            actions.append("train + \nbuild cache")
            actions.append("train\nfrom cache")

            results_table = ["Scenario|Runtime (seconds)|Memory (MB)|CPU %", "----|----|----|----"]
            if benchmark_vw:
                vw_train_no_cache_benchmark_means, vw_train_no_cache_benchmark_stds = benchmark_cmd(vw_train_cmd, "vw", times, vw_clean_cache)
                results_table.append(format_metrics_row("vw train, no cache", vw_train_no_cache_benchmark_means, vw_train_no_cache_benchmark_stds))
                vw_time_values.append(vw_train_no_cache_benchmark_means[0])
                vw_mem_values.append(vw_train_no_cache_benchmark_means[1] / 1024.)
                vw_cpu_values.append(vw_train_no_cache_benchmark_means[2])

            if benchmark_fw:
                fw_train_no_cache_benchmark_means, fw_train_no_cache_benchmark_stds = benchmark_cmd(fw_train_cmd, "fw", times, fw_clean_cache)
                results_table.append(format_metrics_row("fw train, no cache", fw_train_no_cache_benchmark_means, fw_train_no_cache_benchmark_stds))
                fw_time_values.append(fw_train_no_cache_benchmark_means[0])
                fw_mem_values.append(fw_train_no_cache_benchmark_means[1] / 1024.)
                fw_cpu_values.append(fw_train_no_cache_benchmark_means[2])

            if benchmark_vw:
                vw_train_with_cache_benchmark_means, vw_train_with_cache_benchmark_stds = benchmark_cmd(vw_train_cmd, "vw", times)
                results_table.append(format_metrics_row("vw train, using cache", vw_train_with_cache_benchmark_means, vw_train_with_cache_benchmark_stds))
                vw_time_values.append(vw_train_with_cache_benchmark_means[0])
                vw_mem_values.append(vw_train_with_cache_benchmark_means[1] / 1024.)
                vw_cpu_values.append(vw_train_with_cache_benchmark_means[2])

            if benchmark_fw:
                fw_train_with_cache_benchmark_means, fw_train_with_cache_benchmark_stds = benchmark_cmd(fw_train_cmd, "fw", times)
                results_table.append(format_metrics_row("fw train, using cache", fw_train_with_cache_benchmark_means, fw_train_with_cache_benchmark_stds))
                fw_time_values.append(fw_train_with_cache_benchmark_means[0])
                fw_mem_values.append(fw_train_with_cache_benchmark_means[1] / 1024.)
                fw_cpu_values.append(fw_train_with_cache_benchmark_means[2])

        if action == "predict" or action == "train+predict" or action == "all":
            actions.append("predict,\nno cache")
            if benchmark_vw:
                vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds = benchmark_cmd(vw_predict_cmd, "vw", times)
                vw_model_loss = calc_loss("vw_preds.out", "easy.vw")
                results_table.append(format_metrics_row("vw predict, no cache", vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds))
                vw_time_values.append(vw_predict_no_cache_benchmark_means[0])
                vw_mem_values.append(vw_predict_no_cache_benchmark_means[1] / 1024.)
                vw_cpu_values.append(vw_predict_no_cache_benchmark_means[2])

            if benchmark_fw:
                fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds = benchmark_cmd(fw_predict_cmd, "fw", times)
                fw_model_loss = calc_loss("fw_preds.out", "easy.vw")
                results_table.append(format_metrics_row("fw predict, no cache", fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds))
                fw_time_values.append(fw_predict_no_cache_benchmark_means[0])
                fw_mem_values.append(fw_predict_no_cache_benchmark_means[1] / 1024.)
                fw_cpu_values.append(fw_predict_no_cache_benchmark_means[2])

        print("### Summary")
        print(f"here are the results for {times} runs for each scenario, taking mean and standard deviation values:\n")

        plot_file_name = plot_results(vw_mem_values, fw_mem_values, vw_cpu_values, fw_cpu_values)
        print(f"![benchmark results]({plot_file_name})")

        print("### The numbers")
        for line in results_table:
            print(line)

        print("\n")

        print("### Model equivalence")
        print("see here the loss value calculated over the test predictions for the tested models:")
        if action == "predict" or action == "train+predict" or action == "all":
            print("```")
            if benchmark_vw:
                print(f"Vowpal Wabbit predictions loss: {vw_model_loss}")

            if benchmark_fw:
                print(f"Fwumious Wabbit predictions loss: {fw_model_loss}")
            print("```")

        print("\n")

        print("for more details on what makes Fwumious Wabbit so fast, see [here](https://github.com/outbrain/fwumious_wabbit/blob/benchmark/SPEED.md)")


