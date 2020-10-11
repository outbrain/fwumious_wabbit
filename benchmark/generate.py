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


def generate(train_examples=10000000, test_examples=10000000, num_animals=100000, num_foods=100000, block_beyond=50000):
    i = 0
    f = open("train.vw", "w")
    while i < train_examples:
        animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
        food_type = random.choices(['Plant', 'Meat'])[0]
        missone = random.randint(0, 1)
        if missone:
            animal = random.randint(0, num_animals)
            food = random.randint(0, block_beyond)
        else:
            animal = random.randint(0, block_beyond)
            food = random.randint(0, num_foods)
        f.write(render_example((animal_type, animal), (food_type, food)))
        i += 1

    i = 0
    # this has the same distribution as for train...
    f = open("easy.vw", "w")
    while i < test_examples:
        animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
        food_type = random.choices(['Plant', 'Meat'])[0]
        missone = random.randint(0, 1)
        if missone:
            person = random.randint(0, num_animals)
            movie = random.randint(0, block_beyond)
        else:
            person = random.randint(0, block_beyond)
            movie = random.randint(0, num_foods)
        f.write(render_example((animal_type, person), (food_type, movie)))
        i += 1

        # now we will test for completely unseen combos
    f = open("hard.vw", "w")
    i = 0
    while i < test_examples:
        animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
        food_type = random.choices(['Plant', 'Meat'])[0]
        person = random.randint(block_beyond + 1, num_animals)
        movie = random.randint(block_beyond + 1, num_foods)

        f.write(render_example((animal_type, person), (food_type, movie)))
        i += 1


if __name__ == "__main__":
    dataset_size = 5000000
    if len(sys.argv) == 2:
        dataset_size = int(sys.argv[1])

    generate(dataset_size)
