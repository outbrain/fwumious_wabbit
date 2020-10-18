import random
import sys
from pathlib import Path

# deterministic random seed
random.seed(1)

HERBIVORE = 1
CARNIVORE = 2
PLANT = 100
MEAT = 101

ADICT = {
    HERBIVORE: "Herbivore",
    CARNIVORE: "Carnivore"
}

BDICT = {
    PLANT: "Plant",
    MEAT: "Meat"
}


def get_score(a, b):
    if a[0] == HERBIVORE and b[0] == PLANT:
        return 1
    elif a[0] == CARNIVORE and b[0] == MEAT:
        return 1
    else:
        return -1


def random_features(num_random_features):
    l = [""]
    for x in range(num_random_features):
        namespace = chr(ord('C') + x)
        l.append("|" + namespace + " " + namespace + str(random.randint(0, 10000)))
    return " ".join(l)


def render_example(a, b, num_random_features):
    score = get_score(a, b)
    return " ".join([str(score), u"|A", ADICT[a[0]] + u"-" + str(a[1]), u"|B",
                     BDICT[b[0]] + u"-" + str(b[1])]) + random_features(num_random_features) + "\n"


def generate(output_dir, train_examples, test_examples, feature_variety, num_random_features):
    f = open(output_dir / "vw_namespace_map.csv", "w");
    f.write("A,animal\n")
    f.write("B,food\n")
    for x in range(num_random_features):
        namespace = chr(ord('C') + x)
        f.write(namespace + ",somefeature\n")

    i = 0
    f = open(output_dir / "train.vw", "w")
    block_beyond = int(feature_variety / 4.0)
    while i < train_examples:
        add_dataset_record(f, block_beyond, feature_variety, num_random_features)
        i += 1

    i = 0
    # this has the same distribution as for train...
    f = open(output_dir / "easy.vw", "w")
    while i < test_examples:
        add_dataset_record(f, block_beyond, feature_variety, num_random_features)
        i += 1

    # now we will test for completely unseen combos
    f = open(output_dir / "hard.vw", "w")
    i = 0
    while i < test_examples:
        animal_type = random.choices([HERBIVORE, CARNIVORE])[0]
        food_type = random.choices([PLANT, MEAT])[0]
        animal_name = random.randint(block_beyond + 1, feature_variety)
        food_name = random.randint(block_beyond + 1, feature_variety)

        f.write(render_example((animal_type, animal_name), (food_type, food_name), num_random_features))
        i += 1


def add_dataset_record(f, block_beyond, feature_variety, num_random_features):
    animal_type = random.choices([HERBIVORE, CARNIVORE])[0]
    food_type = random.choices([PLANT, MEAT])[0]
    missone = random.randint(0, 1)
    if missone:
        animal_name = random.randint(0, feature_variety)
        food_name = random.randint(0, block_beyond)
    else:
        animal_name = random.randint(0, block_beyond)
        food_name = random.randint(0, feature_variety)
    f.write(render_example((animal_type, animal_name), (food_type, food_name), num_random_features))


if __name__ == "__main__":
    dataset_size = 500000
    if len(sys.argv) == 2:
        dataset_size = int(sys.argv[1])

    generate(Path(""), dataset_size, dataset_size, 1000, 10)
