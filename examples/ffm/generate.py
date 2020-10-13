import random, pathlib
DATASETS_DIRECTORY = pathlib.Path("datasets")
TRAIN_EXAMPLES = 100000
EVAL_EXAMPLES = 10000
NUM_ANIMALS = 5
NUM_FOODS = 5
BLOCK_BEYOND = 3

Aval = []
Bval = []

random.seed(1)

def get_score(a,b):
    if (a[0] == "Herbivore" and b[0] == "Plant"):
        return 1
    elif a[0] == "Carnivore" and b[0] == "Meat":
        return 1
    else:
        return -1
    
def render_example(a, b):
    score = get_score(a, b);
    return " ".join([str(score), u"|A", a[0] + u"-" + str(a[1]), u"|B", b[0] + u"-" + str(b[1])]) + "\n"


DATASETS_DIRECTORY.mkdir(exist_ok=True);

f = open(DATASETS_DIRECTORY / "vw_namespace_map.csv", "w");
f.write("A,animal\n")
f.write("B,food\n")

i = 0 
f = open(DATASETS_DIRECTORY / "train.vw", "w")
while i < TRAIN_EXAMPLES:
    animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
    food_type = random.choices(['Plant', 'Meat'])[0]
    missone = random.randint(0,1)
    if missone:
        person = random.randint(0, NUM_ANIMALS)
        movie = random.randint(0, BLOCK_BEYOND)
    else:
        person = random.randint(0, BLOCK_BEYOND)
        movie = random.randint(0, NUM_FOODS)
    f.write(render_example((animal_type, person), (food_type, movie)))
    i+=1 

i = 0
# this has the same distribution as for train...
f = open(DATASETS_DIRECTORY /"test-easy.vw", "w")
while i < EVAL_EXAMPLES:
    animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
    food_type = random.choices(['Plant', 'Meat'])[0]
    missone = random.randint(0,1)
    if missone:
        person = random.randint(0, NUM_ANIMALS)
        movie = random.randint(0, BLOCK_BEYOND)
    else:
        person = random.randint(0, BLOCK_BEYOND)
        movie = random.randint(0, NUM_FOODS)
    f.write(render_example((animal_type, person), (food_type, movie)))
    i+=1 

# now we will test for completely unseen combos
f = open(DATASETS_DIRECTORY /"test-hard.vw", "w")
i = 0
while i < EVAL_EXAMPLES:
    animal_type = random.choices(['Herbivore', 'Carnivore'])[0]
    food_type = random.choices(['Plant', 'Meat'])[0]
    person = random.randint(BLOCK_BEYOND+1, NUM_ANIMALS)
    movie = random.randint(BLOCK_BEYOND+1, NUM_FOODS)

    f.write(render_example((animal_type, person), (food_type, movie)))
    i+=1 


