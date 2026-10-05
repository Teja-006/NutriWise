"""
food_classes.py
----------------
Single source of truth for the 42-class segmentation label mapping.

Previously this dict was copy-pasted into train_food_multiclass.py,
predict_food_multiclass.py, and test_food_multiclass.py separately — if one
copy ever drifted from the others (a renamed/reordered class), predictions
would silently point at the wrong label. Import this everywhere instead.
"""

NUM_CLASSES = 42

CLASSES = {
    0: "background",
    1: "Bottle-gourd-curry",
    2: "aloo-capsicum",
    3: "aloo-curry",
    4: "aloo-fry",
    5: "beans-curry",
    6: "beetroot-kobari",
    7: "beetroot-poriyal",
    8: "bisi-bele-bath",
    9: "boondi",
    10: "cabbage-dry",
    11: "channa-brinjal",
    12: "chicken-dum-biryani",
    13: "chutney",
    14: "curd",
    15: "dondakaya-fry",
    16: "kakarakaya-fry",
    17: "kofta-curry",
    18: "leaf-dal",
    19: "mango-pickle",
    20: "masoor-dal",
    21: "mirchi-ka-salan",
    22: "muddha-pappu",
    23: "non-spicy-curry",
    24: "non-spicy-dal",
    25: "pachi-pulusu",
    26: "papad",
    27: "payasam",
    28: "phulka",
    29: "raita",
    30: "rajma",
    31: "rasam",
    32: "salad",
    33: "sambar",
    34: "steamed-rice",
    35: "tomato-pappu",
    36: "veg-dum-briyani",
    37: "veg-pulao",
    38: "Watermelon",
    39: "Papaya",
    40: "Banana",
    41: "Muskmelon",
}

assert len(CLASSES) == NUM_CLASSES, "CLASSES dict and NUM_CLASSES are out of sync"