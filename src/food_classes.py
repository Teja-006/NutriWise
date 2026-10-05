"""Single source of truth for ITD class names.

Source: ICVGIP 2025 paper "What is there in an Indian Thali?", Table 2.
Mask ID == row number in that table. ID 0 is background.
Verified by sanity checks against the ID galleries and a labeled plate
(IDs 29, 30, 33, 34, 36, 21, 40 matched visually).
IDs 35 and 37 are the least certain, see check notes.
"""

CLASSES = [
    "background",                        # 0
    "aloo-dry-fry",                      # 1
    "avakaya-muddha-pappu-rice",         # 2
    "baby-corn-capsicum-dry",            # 3
    "cabbage-pakodi",                    # 4
    "cabbage-fry",                       # 5
    "capsicum-paneer-curry",             # 6
    "chakar-pongal",                     # 7
    "chole-masala",                      # 8
    "cluster-beans-curry",               # 9
    "cucumber-raita",                    # 10
    "gobi-masala-curry",                 # 11
    "gutti-vankaya-curry",               # 12
    "jeera-rice",                        # 13
    "mixed-curry",                       # 14
    "muskmelon",                         # 15
    "rajma",                             # 16
    "rasgulla",                          # 17
    "sambar",                            # 18
    "tomato-rasam",                      # 19
    "vankaya-ali-karam",                 # 20
    "veg-biriyani",                      # 21
    "aloo-curry",                        # 22
    "curd",                              # 23
    "dal",                               # 24
    "fresh-chutney",                     # 25
    "green-salad",                       # 26
    "moong-beans-curry",                 # 27
    "khichdi",                           # 28
    "lemon-rice",                        # 29
    "live-roti-with-ghee",               # 30
    "non-spicy-curry-bottle-gourd",      # 31
    "papad",                             # 32
    "plain-rice",                        # 33
    "watermelon",                        # 34
    "aloo-fry",                          # 35
    "banana",                            # 36
    "mix-fruit",                         # 37
    "non-spicy-baby-corn-capsicum-dry",  # 38
    "sweet",                             # 39
    "tomato-rice",                       # 40
    "fried-papad-rings",                 # 41
    "gravy",                             # 42
    "ivy-gourd-fry",                     # 43
    "mango-pickle",                      # 44
    "papad-chat",                        # 45
    "pepper-rasam",                      # 46
    "pineapple",                         # 47
    "corn-fry",                          # 48
    "paneer-curry",                      # 49
    "semiya",                            # 50
]

NUM_CLASSES = len(CLASSES)  # 51
assert NUM_CLASSES == 51, f"expected 51 classes, got {NUM_CLASSES}"

# Aliases so older scripts that import different names keep working.
CLASS_NAMES = CLASSES
ID_TO_NAME = {i: n for i, n in enumerate(CLASSES)}
NAME_TO_ID = {n: i for i, n in enumerate(CLASSES)}