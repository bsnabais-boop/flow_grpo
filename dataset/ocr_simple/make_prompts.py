"""Generate a short-text OCR prompt set for SD 1.5 (1-2 short words, quoted, like dataset/ocr).

Test words are disjoint from train words so eval measures generalisation, not memorisation.
Run from the repo root:  python dataset/ocr_simple/make_prompts.py
"""
import os
import random

TRAIN_WORDS = [
    "OPEN", "SALE", "STOP", "EXIT", "HELLO", "CAFE", "BAR", "HOTEL", "PIZZA", "BOOKS", "MUSIC", "LOVE", "HOME",
    "PARK", "BANK", "TAXI", "BUS", "POLICE", "FIRE", "DANGER", "PUSH", "PULL", "FREE", "NEW", "HOT", "COLD",
    "YES", "NO", "WIN", "GO", "SLOW", "DREAM", "PEACE", "HAPPY", "SUMMER", "WINTER", "SPRING", "BAKERY",
    "COFFEE", "TEA", "MILK", "BREAD", "FISH", "BEER", "WINE", "GYM", "SHOP", "MARKET", "ART", "JAZZ",
    "ROCK", "STAR", "MOON", "SUN", "RAIN", "SNOW", "WILD", "ZOO", "FARM", "LAB",
    "GOOD LUCK", "THANK YOU", "BIG SALE", "NO PARKING", "OPEN NOW", "HOT DOGS", "ICE CREAM", "BEST DAD",
]
TEST_WORDS = [
    "CLOSED", "ENTER", "PHARMACY", "DINER", "TACOS", "RADIO", "SMILE", "OCEAN", "FOREST", "TICKETS",
    "BURGER", "LIBRARY", "KEEP OUT", "WELCOME", "SUSHI", "GARAGE", "CINEMA", "BOXING", "DONUTS", "CANDY",
    "SALE NOW", "TOP SECRET", "GAME OVER", "LOVE YOU",
]
TEMPLATES = [
    'A neon sign that says "{w}" glowing on a brick wall at night',
    'A wooden shop sign with the word "{w}" painted in white letters',
    'A red stop-style road sign with the text "{w}"',
    'A chalkboard outside a cafe with "{w}" written in chalk',
    'A t-shirt with the word "{w}" printed in bold black letters',
    'A coffee mug with "{w}" written on it, on a kitchen table',
    'A paper banner that reads "{w}" hanging in a classroom',
    'A billboard on a highway displaying "{w}" in huge letters',
    'A birthday cake with "{w}" written in icing',
    'A book cover with the title "{w}" in large serif font',
    'A license plate that reads "{w}"',
    'A street sign that says "{w}" against a blue sky',
    'A storefront window with "{w}" in gold letters',
    'A poster with the single word "{w}" in the center',
    'A doormat with the word "{w}" on it',
    'A sticky note with "{w}" handwritten in marker',
]


def make(words, templates, rng):
    return [t.format(w=w) for w in words for t in templates]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    rng = random.Random(0)
    train = make(TRAIN_WORDS, TEMPLATES, rng)
    test = make(TEST_WORDS, TEMPLATES, rng)
    rng.shuffle(train)
    rng.shuffle(test)
    test = test[:128]  # small test set -> quick eval
    for name, rows in [("train.txt", train), ("test.txt", test)]:
        with open(os.path.join(here, name), "w") as f:
            f.write("\n".join(rows) + "\n")
    print(f"train: {len(train)} prompts ({len(TRAIN_WORDS)} words), test: {len(test)} prompts ({len(TEST_WORDS)} words)")
