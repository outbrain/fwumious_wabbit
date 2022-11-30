##########################################################################################
# A script for generation of synthetic data, suitable for sanity-checking a given binary #
##########################################################################################

import random
import pathlib
import os
import argparse


def get_score(a, b):

    if (a[0] == "Herbivore" and b[0] == "Plant"):
        return 1
    elif a[0] == "Carnivore" and b[0] == "Meat":
        return 1
    else:
        return -1


def render_example(a, b):

    score = get_score(a, b)
    return " ".join([
        str(score), u"|A", a[0] + u"-" + str(a[1]), u"|B",
        b[0] + u"-" + str(b[1])
    ]) + "\n"


def generate_synthetic_dataset():

    DATASETS_DIRECTORY.mkdir(exist_ok=True)

    f = open(os.path.join(DATASETS_DIRECTORY, "vw_namespace_map.csv"), "w")
    f.write("A,animal\n")
    f.write("B,food\n")

    i = 0
    f = open(os.path.join(DATASETS_DIRECTORY, "train.vw"), "w")
    while i < TRAIN_EXAMPLES:
        animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
        food_type = random.choices(['Plant', 'Meat'])[0]
        missone = random.randint(0, 1)
        if missone:
            person = random.randint(0, NUM_ANIMALS)
            movie = random.randint(0, BLOCK_BEYOND)
        else:
            person = random.randint(0, BLOCK_BEYOND)
            movie = random.randint(0, NUM_FOODS)
        f.write(render_example((animal_type, person), (food_type, movie)))
        i += 1

    i = 0
    # this has the same distribution as for train...
    f = open(os.path.join(DATASETS_DIRECTORY, "test-easy.vw"), "w")
    while i < EVAL_EXAMPLES:
        animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
        food_type = random.choices(['Plant', 'Meat'])[0]
        missone = random.randint(0, 1)
        if missone:
            person = random.randint(0, NUM_ANIMALS)
            movie = random.randint(0, BLOCK_BEYOND)
        else:
            person = random.randint(0, BLOCK_BEYOND)
            movie = random.randint(0, NUM_FOODS)
        f.write(render_example((animal_type, person), (food_type, movie)))
        i += 1

    # now we will test for completely unseen combos
    f = open(os.path.join(DATASETS_DIRECTORY, "test-hard.vw"), "w")
    i = 0
    while i < EVAL_EXAMPLES:
        animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
        food_type = random.choices(['Plant', 'Meat'])[0]
        person = random.randint(BLOCK_BEYOND + 1, NUM_ANIMALS)
        movie = random.randint(BLOCK_BEYOND + 1, NUM_FOODS)

        f.write(render_example((animal_type, person), (food_type, movie)))
        i += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FW synthetic data generator.")
    parser.add_argument("--num_train_examples",
                        type=int,
                        default=1000000,
                        help="How many instances to generate?")
    parser.add_argument("--num_eval_examples",
                        type=int,
                        default=10000,
                        help="How many instances to evaluate?")
    parser.add_argument("--num_animals",
                        type=int,
                        default=5,
                        help="How many possible animals are there?")
    parser.add_argument("--num_foods",
                        type=int,
                        default=5,
                        help="Number of possible foods for the animals?")
    parser.add_argument("--block_beyond",
                        type=int,
                        default=3,
                        help="block_beyond parameter.")
    parser.add_argument("--random_seed",
                        type=int,
                        default=1,
                        help="Random seed for the generation.")

    args = parser.parse_args()

    DATASETS_DIRECTORY = pathlib.Path("datasets")
    TRAIN_EXAMPLES = args.num_train_examples
    EVAL_EXAMPLES = args.num_eval_examples
    NUM_ANIMALS = args.num_animals
    NUM_FOODS = args.num_foods
    BLOCK_BEYOND = args.block_beyond
    Aval, Bval = [], []

    random.seed(args.random_seed)

    generate_synthetic_dataset()
