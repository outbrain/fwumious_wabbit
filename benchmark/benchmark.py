import math
import os.path
import subprocess
import sys
import psutil
import platform
import generate
import gzip
import shutil
from measure import measure
from calc_loss import calc_loss
from pathlib import Path
from sys import platform as sys_platform

debug = False
VW = "vw" 
FW = "../target/release/fw"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def rm_quietly(f):
    if os.path.isfile(f):
        os.remove(f)


def cleanup():
    rm_quietly("work_dir/train.vw")
    rm_quietly("work_dir/train.vw.gz")
    rm_quietly("work_dir/easy.vw")
    rm_quietly("work_dir/hard.vw")
    vw_clean_cache()
    fw_clean_cache()


def vw_clean_cache():
    rm_quietly("work_dir/train.vw.gz.cache")
    rm_quietly("work_dir/easy.vw.gz.cache")
    rm_quietly("work_dir/train.vw.cache")
    rm_quietly("work_dir/easy.vw.cache")


def fw_clean_cache():
    rm_quietly("work_dir/train.vw.gz.fwcache")
    rm_quietly("work_dir/easy.vw.gz.fwcache")
    rm_quietly("work_dir/train.vw.fwcache")
    rm_quietly("work_dir/easy.vw.fwcache")


def print_system_info():
    rprint("### CPU Info")

    # number of cores
    # "Physical cores: %i" % psutil.cpu_count(logical=False)
    # "Total cores: %i" % psutil.cpu_count(logical=True)
    uname = platform.uname()

    if sys_platform == "darwin":
        rprint(f"""```
Current Frequency: {psutil.cpu_freq().current:.2f}Mhz
Machine: {uname.machine}
Machine: {uname.machine}
Processor: {uname.processor}```
""")
    else:
        cpu_info_cmd = "cat /proc/cpuinfo  | grep \"model name\" | awk -F\": \" '{print $2}' | head -n 1"
        cpu_info = subprocess.check_output(cpu_info_cmd, shell=True).decode('utf-8').strip("\n")

        rprint(f"""```
{cpu_info}
```""")

    rprint(f"""### Operating System
```
System: {uname.system}
Version: {uname.version}
```""")


def gzip_file(f):
    with open(f, 'rb') as f_in, gzip.open(f + ".gz", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def time_bash_cmd(cmd, proc_name):
    return measure(cmd, proc_name)


def benchmark_cmd(cmd, proc_name, times = 1, run_before=None):
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


def print_prerequisites_and_running():
    rprint(f"""## Prerequisites and running
you should have Vowpal Wabbit installed, as the benchmark invokes it via the 'vw' command.

additionally the rust toolchain (particularly cargo and rustc) is required in order to build Fwumious Wabbit (the benchmark invokes '../target/release/fw') 
in order to build and run the benchmark use one of these bash scripts:
```
./run_with_plots.sh
```
in order to run the benchmark and plot the results (requires matplotlib, last used with version 2.1.2)
or, if you just want the numbers with less dependencies run:
```
./run_without_plots.sh
```
## Latest run setup

### versions:
```
vowpal wabbit {vowpal_wabbit_version}
{fwumious_wabbit_version} (git commit: {fwumious_wabbit_revision})
```
""")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("""syntax: benchmark fw|vw|all cleanup|generate|train|predict|train+predict|all True|False
    first argument: evaluate fwumious wabbit, vowpal wabbit or both
    second argument - step to run:
        cleanup - delete intermediate files
        generate - generate synthetic dataset
        train - train a model with and without cache
        predict - use the trained model from the train step to make predictions
        train+predict - run both steps
        all - run all steps
    third argument - plot results using matplotlib (True|False)
    """)
#        ffm - compare fwumious wabbit field aware factorization machine vs. fwumious wabbit logistic regression


        exit()

    if not os.path.isdir("work_dir"):
        os.mkdir("work_dir")

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

    BENCHMARK_MD_FILEPATH = "work_dir/BENCHMARK.new.md"

    with open(BENCHMARK_MD_FILEPATH, "w") as readme:
        def rprint(*args, **kwargs):
            print(*args, file=readme, **kwargs)
        rprint("")  # Clear file

        params = "-l 0.1 -b 25 --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all"
        interactions = "--interactions AB --keep A --keep B --keep C --keep D --keep E --keep F --keep G --keep H --keep I --keep J --keep K --keep L"

        vw_train_cmd = f"{VW} --data work_dir/train.vw -p work_dir/vw_train_preds.out --final_regressor work_dir/vw_model --save_resume {params} {interactions}"
        fw_train_cmd = f"{FW} --data work_dir/train.vw -p work_dir/fw_train_preds.out --final_regressor work_dir/fw_model --save_resume {params} {interactions}"

        vw_predict_cmd = f"{VW} -t --data work_dir/easy.vw -p work_dir/vw_easy_preds.out --initial_regressor work_dir/vw_model {params} {interactions}"
        fw_predict_cmd = f"{FW} -t --data work_dir/easy.vw -p work_dir/fw_easy_preds.out --initial_regressor work_dir/fw_model {params} {interactions}"

        if action in ["cleanup", "generate", "all"]:
            cleanup()

        train_examples = 1_000_000
        test_examples = 1_000_000
        feature_variety = 1000
        num_random_features = 10

        if action in ["generate", "all"]:
            eprint(f"Generating test data, training examples: {train_examples}, test examples: {test_examples}")
            generate.generate(Path("work_dir"), train_examples, test_examples, feature_variety, num_random_features)
            # Currently we don't benchmark over gzip files as it's just one more layer of complication
            #gzip_file("work_dir/train.vw")

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

        hard_results_table = ["Scenario|Runtime (seconds)|Memory (MB)|CPU %", "----|----:|----:|----:"]

        if action in ["train", "predict", "train+predict", "all"]:
            rprint("""## Scenarios
1. train a new model from a dataset and an output model file - *typical scenario for one-off training on the dataset*
1. train a new model from a cached dataset, and generate an output model - *this is also a typical scenario - we usually run many concurrent model evaluations as part of the model search*
1. use a generated model to make predictions over a dataset read from a text file, and print them to an output predictions file - *this is to illustrate potential serving performance, we don't usually predict from file input as our offline flows always apply online learning. note that when running as daemon we use half as much memory since gradients are not loaded - only model weights.*
""")

        if action in ["train", "train+predict", "all"]:
            actions.append("train,\nno cache")
            actions.append("train\nfrom cache")

            results_table = ["Scenario|Runtime (seconds)|Memory (MB)|CPU %", "----|----:|----:|----:"]

            if benchmark_vw:
                eprint("Measuring vw training without cache nor creating cache") 
                vw_train_no_cache_benchmark_means, vw_train_no_cache_benchmark_stds = benchmark_cmd(vw_train_cmd, "vw", times, vw_clean_cache)
                results_table.append(format_metrics_row("vw train, no cache", vw_train_no_cache_benchmark_means, vw_train_no_cache_benchmark_stds))
                vw_time_values.append(vw_train_no_cache_benchmark_means[0])
                vw_mem_values.append(vw_train_no_cache_benchmark_means[1] / 1024.)
                vw_cpu_values.append(vw_train_no_cache_benchmark_means[2])

            if benchmark_fw:
                eprint("Measuring fw training without cache nor creating cache") 
                fw_train_no_cache_benchmark_means, fw_train_no_cache_benchmark_stds = benchmark_cmd(fw_train_cmd, "fw", times, fw_clean_cache)
                results_table.append(format_metrics_row("fw train, no cache", fw_train_no_cache_benchmark_means, fw_train_no_cache_benchmark_stds))
                fw_time_values.append(fw_train_no_cache_benchmark_means[0])
                fw_mem_values.append(fw_train_no_cache_benchmark_means[1] / 1024.)
                fw_cpu_values.append(fw_train_no_cache_benchmark_means[2])

            if benchmark_vw:
                eprint("Creating cache for vw training") 
                # first create cache
                vw_train_cache_cmd = vw_train_cmd + " -c"
                benchmark_cmd(vw_train_cache_cmd, "vw")
                # then benchmark
                eprint("Measuring vw training from cache")
                vw_train_with_cache_benchmark_means, vw_train_with_cache_benchmark_stds = benchmark_cmd(vw_train_cache_cmd, "vw", times)
                results_table.append(format_metrics_row("vw train, using cache", vw_train_with_cache_benchmark_means, vw_train_with_cache_benchmark_stds))
                vw_time_values.append(vw_train_with_cache_benchmark_means[0])
                vw_mem_values.append(vw_train_with_cache_benchmark_means[1] / 1024.)
                vw_cpu_values.append(vw_train_with_cache_benchmark_means[2])

            if benchmark_fw:
                eprint("Creating cache for fw training") 
                # first create cache
                fw_train_cache_cmd = fw_train_cmd + " -c"
                benchmark_cmd(fw_train_cache_cmd, "fw")
                # then train
                eprint("Measuring fw training from cache")
                fw_train_with_cache_benchmark_means, fw_train_with_cache_benchmark_stds = benchmark_cmd(fw_train_cache_cmd, "fw", times)
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
                eprint("Measuring vw prediction, no caches used or created")
                vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds = benchmark_cmd(vw_predict_cmd, "vw", times)
                vw_model_loss = calc_loss("work_dir/vw_easy_preds.out", "work_dir/easy.vw")
                results_table.append(format_metrics_row("vw predict, no cache", vw_predict_no_cache_benchmark_means, vw_predict_no_cache_benchmark_stds))
                vw_time_values.append(vw_predict_no_cache_benchmark_means[0])
                vw_mem_values.append(vw_predict_no_cache_benchmark_means[1] / 1024.)
                vw_cpu_values.append(vw_predict_no_cache_benchmark_means[2])

            if benchmark_fw:
                eprint("Measuring fw prediction only, no caches used or created")
                fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds = benchmark_cmd(fw_predict_cmd, "fw", times)
                fw_model_loss = calc_loss("work_dir/fw_easy_preds.out", "work_dir/easy.vw")
                results_table.append(format_metrics_row("fw predict, no cache", fw_predict_no_cache_benchmark_means, fw_predict_no_cache_benchmark_stds))
                fw_time_values.append(fw_predict_no_cache_benchmark_means[0])
                fw_mem_values.append(fw_predict_no_cache_benchmark_means[1] / 1024.)
                fw_cpu_values.append(fw_predict_no_cache_benchmark_means[2])


        rprint(f"""
## Model
We train a logistic regression model, applying online learning one example at a time (no batches), 
using '--adaptive' flag for adaptive learning rates (AdaGrad variant).

## Results
here are the results for {times} runs for each scenario, taking mean values:""")

        if use_plots and action in ["all", "train", "predict", "train+predict"]:
            plot_file_name = "benchmark_results.png"
            left_label = "Vowpal Wabbit" if benchmark_vw else None
            right_label = "Fwumious Wabbit" if benchmark_vw else None
            plot_results(f"work_dir/{plot_file_name}", left_label, right_label, actions, vw_time_values, fw_time_values, vw_mem_values, fw_mem_values, vw_cpu_values, fw_cpu_values)
            rprint(f"![benchmark results]({plot_file_name})")

        if action in ["train", "predict", "train+predict", "all"]:
            for line in results_table:
                rprint(line)
                print(line)
        print("\n")
        rprint("""
### Model equivalence
loss values for the test set:
""")
        if action in ["predict", "train+predict", "all"]:
            rprint("```")
            if benchmark_vw:
                rprint(f"Vowpal Wabbit predictions loss: {vw_model_loss:.4f}")
                print(f"Vowpal Wabbit predictions loss: {vw_model_loss:.4f}")

            if benchmark_fw:
                rprint(f"Fwumious Wabbit predictions loss: {fw_model_loss:.4f}")
                print(f"Fwumious Wabbit predictions loss: {fw_model_loss:.4f}")

            rprint("```")

        rprint("\n")

        rprint("for more details on what makes Fwumious Wabbit so fast, see [here](https://github.com/outbrain/fwumious_wabbit/blob/benchmark/SPEED.md)")

        if action in ["generate", "all"]:
            rprint(f"""
### Dataset details
we generate a synthetic dataset with {train_examples:,} train records ('train.vw'), and {test_examples:,} test records ('easy.vw').
\nthe task is 'Eat-Rate prediction' - each record describes the observed result of a single feeding experiment.
each record is made of a type of animal, a type of food (in Vowpal Wabbit jargon these are our namespaces A and B respectively), and a label indicating whether the animal ate the food.
the underlying model is simple - animals are either herbivores or carnivores,
and food is either plant based or meat based.\n
herbivores always eat plants (and only plants), and carnivores always eat meat (and only meat).\n
we name animals conveniently using the pattern 'diet-id', for example 'Herbivore-1234' and 'Carnivore-5678'
and the food similarly as 'food_type-id' - for example 'Plant-678' and 'Meat-234' so the expected label for a record is always obvious.
there are {feature_variety:,} animal types, and {feature_variety:,} food types. we generate additional {num_random_features} random features,
to make the dataset dimensions a bit more realistic.
""")

            with open("work_dir/train.vw", "r") as dataset:
                random_features_titles = ""
                random_features_header = ""
                for i in range(min(num_random_features, 6)):
                    random_features_titles = f"{random_features_titles}|feat_{i+2}"
                    random_features_header = f"{random_features_header}|----"

                header_suffix = ""
                if num_random_features > 6:
                    random_features_titles = f"{random_features_titles}|..."
                    random_features_header = f"{random_features_header}|----"
                    header_suffix = "|..."

                rprint(f"""see for example the first 5 lines from the train dataset (after some pretty-printing):

label|animal|food{random_features_titles}
----:|------|----{random_features_header}""")
                for _ in range(5):
                    rprint("|".join(next(dataset).strip("\n").split("|")[0:9]) + header_suffix)

                rprint("\n")

        if action in ["train", "predict", "train+predict", "all"]:
            rprint("""### Feature engineering
if we train using separate 'animal type' and 'food type' features, the model won't learn well, 
since knowing the animal identity alone isn't enough to predict if it will eat or not - and the same 
goes for knowing the food type alone.
so we apply an interaction between the animal type and food type fields.
            """)

        if action in ["ffm", "all"] and False:  # "soft" comment out until I move the output to a separate document
            rprint("""## Field aware factorization machines
in this experiment we demonstrate how field aware factorization machines (FFMs) can better capture 
feature interactions, resulting in better model accuracy.

### Dataset
In the train set we generated, the animals and foods are each divided to two groups - we'll mark them A1 and A2 for the animals,
and F1 and F2 for the foods.
the train set and the test set named 'easy.vw' (used in the previous section) are both drawn from the same distribution, 
with records which belong to {(A1 U A2, F1)} U {(A1, F1 U F2}).
the test set we use here, 'hard.vw', is different: it contains exclusively records from {(A2, F2)} - combinations unseen in the train set.
In order for a model to make correct predictions on this dataset after training on the train dataset, 
it must be able to generalize for unseen combinations.
""")

            fw_hard_ffm_time_values = []
            fw_hard_ffm_mem_values = []
            fw_hard_ffm_cpu_values = []

            ffm_actions = ["train,\nno cache", "predict,\nno cache"]

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

            rprint("we skip the latency, memory and CPU comparison as for this synthetic dataset the difference is negligible.")

            # if use_plots:
            #     ffm_plot_file_name = "ffm_benchmark_results.png"
            #     plot_results(ffm_plot_file_name, "Fwumious Wabbit LR", "Fwumious Wabbit FFM", ffm_actions, fw_hard_time_values, fw_hard_ffm_time_values, fw_hard_mem_values, fw_hard_ffm_mem_values, fw_hard_cpu_values, fw_hard_ffm_cpu_values)
            #     print(f"![FFM benchmark results]({ffm_plot_file_name})")

            fw_model_loss = calc_loss("work_dir/fw_hard_preds.out", "hard.vw")
            fw_ffm_model_loss = calc_loss("work_dir/fw_ffm_hard_preds.out", "hard.vw")

            rprint(f"""### Loss on the test set")
```
Fwumious Wabbit Logistic Regression predictions loss: {fw_model_loss:.4f}
Fwumious Wabbit FFM predictions loss: {fw_ffm_model_loss:.4f}
```
""")
        vowpal_wabbit_version = None
        if benchmark_vw:
            vowpal_wabbit_version = subprocess.check_output(f"{VW} --version", shell=True).decode('utf-8').strip("\n")
        fwumious_wabbit_version = subprocess.check_output(f"{FW} --version", shell=True).decode('utf-8').strip("\n")
        fwumious_wabbit_revision = subprocess.check_output("git log | head -n 1", shell=True).decode('utf-8').split(" ")[1][0:7]

        print_prerequisites_and_running()

        print_system_info()

    if os.path.isfile(BENCHMARK_MD_FILEPATH):
        shutil.copyfile(BENCHMARK_MD_FILEPATH, "../BENCHMARK.md")

    if os.path.isfile("work_dir/benchmark_results.png"):
        shutil.copyfile("work_dir/benchmark_results.png", "../benchmark_results.png")
