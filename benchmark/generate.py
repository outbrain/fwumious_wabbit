import random
import sys


def get_score(a, b):
    if a[0] == "Herbivore" and b[0] == "Plant":
        return 1
    elif a[0] == "Carnivore" and b[0] == "Meat":
        return 1
    else:
        return -1


def render_example(a, b):
    score = get_score(a, b)
    #    print(a,b)
    return " ".join([str(score), u"|A", a[0] + u"-" + str(a[1]), u"|B", b[0] + u"-" + str(b[1])]) + "\n"


def generate(train_examples, test_examples, feature_variety):
    i = 0
    f = open("train.vw", "w")
    block_beyond = int(feature_variety / 4.0)
    while i < train_examples:
        add_dataset_record(f, block_beyond, feature_variety)
        i += 1

    i = 0
    # this has the same distribution as for train...
    f = open("easy.vw", "w")
    while i < test_examples:
        add_dataset_record(f, block_beyond, feature_variety)
        i += 1

    # now we will test for completely unseen combos
    f = open("hard.vw", "w")
    i = 0
    while i < test_examples:
        animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
        food_type = random.choices(['Plant', 'Meat'])[0]
        animal_name = random.randint(block_beyond + 1, feature_variety)
        food_name = random.randint(block_beyond + 1, feature_variety)

        f.write(render_example((animal_type, animal_name), (food_type, food_name)))
        i += 1


def add_dataset_record(f, block_beyond, feature_variety):
    animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
    food_type = random.choices(['Plant', 'Meat'])[0]
    missone = random.randint(0, 1)
    if missone:
        animal_name = random.randint(0, feature_variety)
        food_name = random.randint(0, block_beyond)
    else:
        animal_name = random.randint(0, block_beyond)
        food_name = random.randint(0, feature_variety)
    f.write(render_example((animal_type, animal_name), (food_type, food_name)))


if __name__ == "__main__":
    dataset_size = 5000000
    if len(sys.argv) == 2:
        dataset_size = int(sys.argv[1])

    generate(dataset_size, dataset_size, 1000)
