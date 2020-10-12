import math
import os.path
import sys
import psutil
import platform
import generate
import gzip
import shutil
from measure import measure
from calc_loss import calc_loss

debug = False


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


def format_metrics_row(action, means, stds):
    if debug:
        return f"{action}|{means[0]:.2f} ± {stds[0]:.2f} | {means[1]/1024.:.0f} ± {stds[1]/1024.:.2f} | {means[2]:.2f} ± {stds[2]:.2f}"
    else:
        return f"{action}|{means[0]:.2f} | {means[1]/1024.:.0f} | {means[2]:.2f}"


def plot_results(filename, left_label, right_label, actions, vw_time_values, fw_time_values, vw_mem_values, fw_mem_values, vw_cpu_values, fw_cpu_values):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ind = np.arange(max(len(fw_time_values), len(vw_time_values)))
    width = 0.35

    if left_label:
        ax1.bar(ind, vw_time_values, width, label=left_label)
        ax2.bar(ind, vw_mem_values, width, label=left_label)
        ax3.bar(ind, vw_cpu_values, width, label=left_label)
    if right_label:
        ax1.bar(ind + width, fw_time_values, width, label=right_label)
        ax2.bar(ind + width, fw_mem_values, width, label=right_label)
        ax3.bar(ind + width, fw_cpu_values, width, label=right_label)
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
    plt.savefig(filename)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("syntax: benchmark fw|vw|all cleanup|generate|train|predict|train+predict|all True|False")
        print("first argument: evaluate fwumious wabbit, vowpal wabbit or both")
        print("second argument - step to run:")
        print("   cleanup - delete intermediate files")
        print("   generate - generate synthetic dataset")
        print("   train - train a model with and without cache")
        print("   predict - use the trained model from the train step to make predictions")
        print("   train+predict - run both steps")
        print("   ffm - compare fwumious wabbit field aware factorization machine vs. fwumious wabbit logistic regression")
        print("   all - run all steps")
        print("\nthird argument - plot results using matplotlib (True|False)")

        exit()

    first_arg_options = ["fw", "vw", "all"]
    if sys.argv[1] not in first_arg_options:
        first_arg_options_str = "', '".join(first_arg_options)
        print(f"first argument must be one of '{first_arg_options_str}'.")
        exit()

    second_arg_options = ["cleanup", "generate", "train", "predict", "train+predict", "ffm", "all"]
    if sys.argv[2] not in second_arg_options:
        second_arg_options_str = "', '".join(second_arg_options)
        print(f"second argument must be one of '{second_arg_options_str}'.")
        exit()

    third_arg_options = ["True", "False"]
    if sys.argv[3] not in third_arg_options:
        third_arg_options_str = "', '".join(third_arg_options)
        print(f"third argument must be one of '{third_arg_options_str}'")
        exit()

    benchmark_fw = sys.argv[1] == "all" or sys.argv[1] == "fw"
    benchmark_vw = sys.argv[1] == "all" or sys.argv[1] == "vw"
    use_plots = sys.argv[3] == "True"

    action = sys.argv[2]

    with open("README.md", "w") as readme:
        readme.write("")  # clear file
        sys.stdout = readme
        print("## Prerequisites and running")
        print("you should have Vowpal Wabbit installed, as the benchmark invokes it via the 'vw' command.")
        print("additionally the rust compiler is required in order to build Fwumious Wabbit (using './target/release/fw') ")
        print("in order to build and run the benchmark use one of these bash scripts:")
        print("```")
        print("./run_with_plots.sh")
        print("```")
        print("in order to run the benchmark and plot the results (requires matplotlib, last used with version 2.1.2)")
        print("\nor, if you just want the numbers with less dependencies run:")
        print("```")
        print("./run_without_plots.sh")
        print("```\n")
        print("## Latest run setup\n")

        print_system_info()

        vw_train_cmd = "vw --data train.vw.gz -l 0.1 -p blah.out -b 25 -c --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor vw_model --save_resume --interactions AB"
        fw_train_cmd = "../target/release/fw --data train.vw.gz -l 0.1 -p blah.out -b 25 -c --adaptive --fastmath --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor fw_model --save_resume --interactions AB"
        fw_ffm_train_cmd = "../target/release/fw --data train.vw.gz -l 0.1 -p blah.out -b 25 -c --adaptive --fastmath --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor fw_ffm_model --save_resume --keep A --keep B --ffm_k 10 --ffm_field A --ffm_field B"

        vw_predict_cmd = "vw --data easy.vw -t -p vw_easy_preds.out --initial_regressor vw_model --hash all --interactions AB"
        fw_predict_cmd = "../target/release/fw --data easy.vw -t -b 25 -p fw_easy_preds.out --initial_regressor fw_model --hash all --interactions AB"

        fw_predict_hard_cmd = "../target/release/fw --data hard.vw -t -b 25 -p fw_hard_preds.out --initial_regressor fw_model --link logistic --hash all --interactions AB"
        fw_ffm_predict_hard_cmd = "../target/release/fw --data hard.vw -t -b 25 -p fw_ffm_hard_preds.out --initial_regressor fw_ffm_model --hash all --keep A --keep B --ffm_k 10 --ffm_field A --ffm_field B"

        if action in ["cleanup", "generate", "all"]:
            cleanup()

        if action in ["generate", "all"]:
            train_examples = 10000000
            test_examples = 10000000

            feature_variety = 1000

            print("### Dataset details")
            print(f"we generate a synthetic dataset with {train_examples:,} train records ('train.vw'), and {test_examples:,} test records ('easy.vw').\n")
            print("the task is 'Eat-Rate prediction' - each record describes the observed result of a single feeding experiment.\n")
            print("each record is made of a type of animal, a type of food, and a label indicating whether the animal ate the food.\n")
            print("the underlying model is simple - animals are either herbivores or carnivores,")
            print("and food is either plant based or meat based.")
            print("herbivores always eat plants (and only plants), and carnivores always eat meat (and only meat).\n")
            print("we name animals conveniently using the pattern 'diet-id', for example 'Herbivore-1234' and 'Carnivore-5678',")
            print("and the food similarly as 'food_type-id' - for example 'Plant-678'")
            print(" and 'Meat-234' so the expected label for a record is always obvious.\n")
            print(f"there are {feature_variety:,} animal types, and {feature_variety:,} food types.")
            print("\n")

            generate.generate(train_examples, test_examples, feature_variety)
            with open("train.vw", "r") as dataset:
                print("see for example the first 5 lines from the train dataset (after some pretty-printing):")
                print("label|animal|food")
                print("-----|------|----")
                for _ in range(5):
                    print(next(dataset).strip("\n"))
                print("\n")
            gzip_file("train.vw")

        times = 3
        actions = []
        vw_time_values = []
        vw_mem_values = []
        vw_cpu_values = []
        fw_time_values = []
        fw_mem_values = []
        fw_cpu_values = []
        fw_hard_time_values = []
        fw_hard_mem_values = []
        fw_hard_cpu_values = []

        print("## Results\n")
        print("We train a logistic regression model, applying online learning one example at a time (no batches), \n")
        print("using '--adaptive' learning rates (AdaGrad variant).\n")
        print("if we train using separate 'animal type' and 'food type' features, the model won't learn well, ")
        print("since knowing the animal identity alone isn't enough to predict if it will eat or not - and the same ")
        print("goes for knowing the food type alone.\n")
        print("That's why we use an interaction between the animal type and food type.\n")

        hard_results_table = ["Scenario|Runtime (seconds)|Memory (MB)|CPU %", "----|----|----|----"]

        if action in ["train", "train+predict", "all"]:
            print("**we measure 3 scenarios:**")
            print("1. train a new model from a gzipped dataset, generating a gzipped cache file for future runs, and an output model file - *this is a typical scenario in our AutoML system - we start by generating the cache file for the next runs.*")
            print("1. train a new model over the dataset in the gzipped cache, and generate an output model - *this is also a typical scenario - we usually run many concurrent model evaluations as part of the model search*")
            print("1. use a generated model to make predictions over a dataset read from a text file, and print them to an output predictions file - *this is to illustrate potential serving performance, we don't usually predict from file input as our offline flows always apply online learning. note that when running as daemon we use half as much memory since gradients are not loaded - only model weights.*")
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
                hard_results_table.append(format_metrics_row("fw train, using cache", fw_train_with_cache_benchmark_means, fw_train_with_cache_benchmark_stds))
                fw_time_values.append(fw_train_with_cache_benchmark_means[0])
                fw_hard_time_values.append(fw_train_with_cache_benchmark_means[0])
                fw_mem_values.append(fw_train_with_cache_benchmark_means[1] / 1024.)
                fw_hard_mem_values.append(fw_train_with_cache_benchmark_means[1] / 1024.)
                fw_cpu_values.append(fw_train_with_cache_benchmark_means[2])
                fw_hard_cpu_values.append(fw_train_with_cache_benchmark_means[2])

        if action in ["predict", "train+predict", "all"]:
            actions.append("predict,\nno cache")
            if benchmark_vw:
                vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds = benchmark_cmd(vw_predict_cmd, "vw", times)
                vw_model_loss = calc_loss("vw_easy_preds.out", "easy.vw")
                results_table.append(format_metrics_row("vw predict, no cache", vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds))
                vw_time_values.append(vw_predict_no_cache_benchmark_means[0])
                vw_mem_values.append(vw_predict_no_cache_benchmark_means[1] / 1024.)
                vw_cpu_values.append(vw_predict_no_cache_benchmark_means[2])

            if benchmark_fw:
                fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds = benchmark_cmd(fw_predict_cmd, "fw", times)
                fw_model_loss = calc_loss("fw_easy_preds.out", "easy.vw")
                results_table.append(format_metrics_row("fw predict, no cache", fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds))
                fw_time_values.append(fw_predict_no_cache_benchmark_means[0])
                fw_mem_values.append(fw_predict_no_cache_benchmark_means[1] / 1024.)
                fw_cpu_values.append(fw_predict_no_cache_benchmark_means[2])

        print("### Summary")
        print(f"here are the results for {times} runs for each scenario, taking mean values:\n")

        if use_plots and action in ["all", "train", "predict", "train+predict"]:
            plot_file_name = "benchmark_results.png"
            plot_results(plot_file_name, 'Vowpal Wabbit', "Fwumious Wabbit", actions, vw_time_values, fw_time_values, vw_mem_values, fw_mem_values, vw_cpu_values, fw_cpu_values)
            print(f"![benchmark results]({plot_file_name})")

        if action in ["train", "predict", "train+predict", "all"]:
            for line in results_table:
                print(line)

        print("\n")

        print("### Model equivalence")
        print("see here the loss value calculated over the test predictions for the tested models:")
        if action in ["predict", "train+predict", "all"]:
            print("```")
            if benchmark_vw:
                print(f"Vowpal Wabbit predictions loss: {vw_model_loss:.4f}")

            if benchmark_fw:
                print(f"Fwumious Wabbit predictions loss: {fw_model_loss:.4f}")

            print("```")

        print("\n")

        print("for more details on what makes Fwumious Wabbit so fast, see [here](https://github.com/outbrain/fwumious_wabbit/blob/benchmark/SPEED.md)")

        if action in ["ffm", "all"]:
            print("## Field aware factorization machines")
            print("in this experiment we demonstrate how field aware factorization machines (FFMs) can better capture ")
            print("feature interactions, resulting in better model accuracy.\n")
            print("### Dataset")
            print("In the train set we generated, the animals and foods are each divided to two groups - we'll mark them A1 and A2 for the animals,")
            print("and F1 and F2 for the foods.\n")
            print("the train set and the test set named 'easy.vw' (used in the previous section) are both drawn from the same distribution, ")
            print("with records which belong to {(A1 U A2, F1)} U {(A1, F1 U F2}).\n")
            print("the test set we use here, 'hard.vw', is different: it contains exclusively records from {(A2, F2)} - combinations unseen in the train set.\n")
            print("In order for a model to make correct predictions on this dataset after training on the train dataset, ")
            print("it must be able to generalize for unseen combinations.")

            fw_hard_ffm_time_values = []
            fw_hard_ffm_mem_values = []
            fw_hard_ffm_cpu_values = []

            ffm_actions = ["train\nfrom cache", "predict,\nno cache"]

            fw_ffm_train_with_cache_benchmark_means, fw_ffm_train_with_cache_benchmark_stds = benchmark_cmd(fw_train_cmd, "fw", times)
            hard_results_table.append(format_metrics_row("fw FFM train, using cache", fw_ffm_train_with_cache_benchmark_means, fw_train_with_cache_benchmark_stds))
            fw_hard_ffm_time_values.append(fw_ffm_train_with_cache_benchmark_means[0])
            fw_hard_ffm_mem_values.append(fw_ffm_train_with_cache_benchmark_means[1] / 1024.)
            fw_hard_ffm_cpu_values.append(fw_ffm_train_with_cache_benchmark_means[2])

            fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds = benchmark_cmd(fw_predict_hard_cmd, "fw", times)
            hard_results_table.append(format_metrics_row("fw predict, no cache", fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds))
            fw_hard_time_values.append(fw_predict_no_cache_benchmark_means[0])
            fw_hard_mem_values.append(fw_predict_no_cache_benchmark_means[1] / 1024.)
            fw_hard_cpu_values.append(fw_predict_no_cache_benchmark_means[2])

            fw_ffm_predict_no_cache_benchmark_means, fw_ffm_predict_no_cache_benchmark_stds = benchmark_cmd(fw_ffm_predict_hard_cmd, "fw", times)
            hard_results_table.append(format_metrics_row("fw FFM predict, no cache", fw_ffm_predict_no_cache_benchmark_means, fw_ffm_predict_no_cache_benchmark_means))
            fw_hard_ffm_time_values.append(fw_ffm_predict_no_cache_benchmark_means[0])
            fw_hard_ffm_mem_values.append(fw_ffm_predict_no_cache_benchmark_means[1] / 1024.)
            fw_hard_ffm_cpu_values.append(fw_ffm_predict_no_cache_benchmark_means[2])

            print("we skip the latency, memory and CPU comparison as for this synthetic dataset the difference is negligible.")

            # if use_plots:
            #     ffm_plot_file_name = "ffm_benchmark_results.png"
            #     plot_results(ffm_plot_file_name, "Fwumious Wabbit LR", "Fwumious Wabbit FFM", ffm_actions, fw_hard_time_values, fw_hard_ffm_time_values, fw_hard_mem_values, fw_hard_ffm_mem_values, fw_hard_cpu_values, fw_hard_ffm_cpu_values)
            #     print(f"![FFM benchmark results]({ffm_plot_file_name})")

            fw_model_loss = calc_loss("fw_hard_preds.out", "hard.vw")
            fw_ffm_model_loss = calc_loss("fw_ffm_hard_preds.out", "hard.vw")

            print("### Loss on the test set")
            print("```")
            print(f"Fwumious Wabbit Logistic Regression predictions loss: {fw_model_loss:.4f}")
            print(f"Fwumious Wabbit FFM predictions loss: {fw_ffm_model_loss:.4f}")
            print("```")



