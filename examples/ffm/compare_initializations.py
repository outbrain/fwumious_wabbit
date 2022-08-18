# A simple script for initialization quality comparison
from sklearn import metrics
import glob

def extract_ground_truth(ground_truth_file: str):
    """ Get ground truth labels """

    final_labels = []
    with open(ground_truth_file, "rt", encoding="utf-8") as ground:
        for line in ground:
            label = int(line.strip().split()[0])
            if label == 1:
                final_labels.append(1)
            else:
                final_labels.append(0)
    return final_labels


def extract_initialization_related_outputs(outputs_folder: str):

    result_map = dict()
    for output_file in glob.glob(outputs_folder):
        if "initialization_" in output_file:
            output_probabilities = []
            with open(output_file, "rt", encoding="utf-8") as out_file:
                for line in out_file:
                    probability = line.strip()
                    output_probabilities.append(float(probability))
            result_map[output_file] = output_probabilities
    return result_map


if __name__ == "__main__":
    ground_truth_file: str = "./datasets/test-hard.vw"
    ground_truth = extract_ground_truth(ground_truth_file)
    initialization_performances = \
        extract_initialization_related_outputs("./predictions/*")
    for k, v in initialization_performances.items():
        print(k, " -- AUC of: ", metrics.roc_auc_score(ground_truth, v))
