import random

# Lists of adjectives, nouns, and verbs to create funny names
adjectives = [
    "Silly", "Grumpy", "Happy", "Wacky", "Bouncy", "Nervous",
    "Lazy", "Fuzzy", "Giggly", "Sassy", "Witty", "Clumsy"
]

nouns = [
    "Banana", "Penguin", "Pickle", "Unicorn", "Muffin",
    "Taco", "Llama", "Noodle", "Wombat", "Zebra",
    "Giraffe", "Koala", "Cookie", "Donut", "Dinosaur"
]

verbs = [
    "Dancing", "Flying", "Running", "Jumping",
    "Swimming", "Singing", "Laughing", "Wobbling",
    "Skipping", "Rolling", "Hopping"
]

def generate_name():
    # Randomly select one word from each list
    adj = random.choice(adjectives)
    noun = random.choice(nouns)
    verb = random.choice(verbs)

    # Combine them into a funny name
    name = f"{adj} {noun} {verb}"

    return name