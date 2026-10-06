"""
food_nutrition.py
-----------------
Per-100g nutrition for each food class, scaled by estimated grams.

IMPORTANT: every number below is an APPROXIMATE estimate for typical
home-style cooked dishes, written from general knowledge, NOT looked up
in IFCT 2017. Home recipes vary a lot (oil/ghee/sugar). Verify against
IFCT 2017 or another reference before presenting as fact.
Least certain: sweet (39), semiya (50, could be upma or payasam),
gravy (42), mix-fruit (37), avakaya-muddha-pappu-rice (2).

Keys must match food_classes.py exactly. Run `python food_nutrition.py`
to list any class that is missing an entry.
"""

FIELDS = ("calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")


def _n(kcal, protein, carbs, fat, fiber):
    return {"calories_kcal": kcal, "protein_g": protein, "carbs_g": carbs,
            "fat_g": fat, "fiber_g": fiber}


NUTRITION_PER_100G = {
    "aloo-dry-fry":                     _n(150, 2.2, 20.0, 7.0, 2.0),
    "avakaya-muddha-pappu-rice":        _n(150, 4.0, 25.0, 3.5, 2.0),
    "baby-corn-capsicum-dry":           _n(90,  2.5, 9.0,  5.0, 2.5),
    "cabbage-pakodi":                   _n(260, 8.0, 24.0, 15.0, 4.0),
    "cabbage-fry":                      _n(80,  2.0, 8.0,  4.5, 2.5),
    "capsicum-paneer-curry":            _n(160, 7.0, 7.0,  12.0, 1.5),
    "chakar-pongal":                    _n(190, 3.5, 33.0, 5.0, 1.0),
    "chole-masala":                     _n(150, 7.0, 20.0, 5.0, 6.0),
    "cluster-beans-curry":              _n(85,  3.0, 9.0,  4.0, 3.5),
    "cucumber-raita":                   _n(45,  2.5, 4.0,  2.0, 0.5),
    "gobi-masala-curry":                _n(95,  3.0, 8.0,  6.0, 2.5),
    "gutti-vankaya-curry":              _n(120, 3.0, 8.0,  8.5, 3.0),
    "jeera-rice":                       _n(150, 3.0, 28.0, 3.0, 0.5),
    "mixed-curry":                      _n(90,  2.5, 10.0, 4.5, 3.0),
    "muskmelon":                        _n(34,  0.8, 8.0,  0.2, 0.9),
    "rajma":                            _n(120, 6.0, 15.0, 3.5, 5.0),
    "rasgulla":                         _n(185, 4.0, 40.0, 1.0, 0.0),
    "sambar":                           _n(55,  2.7, 7.5,  1.7, 1.8),
    "tomato-rasam":                     _n(25,  0.8, 4.0,  0.7, 0.5),
    "vankaya-ali-karam":                _n(100, 2.0, 8.0,  7.0, 3.0),
    "veg-biriyani":                     _n(150, 3.5, 24.0, 4.5, 1.5),
    "aloo-curry":                       _n(95,  2.0, 14.0, 3.5, 1.8),
    "curd":                             _n(60,  3.1, 4.7,  3.3, 0.0),
    "dal":                              _n(100, 5.5, 13.0, 2.8, 3.0),
    "fresh-chutney":                    _n(150, 2.5, 8.0,  12.0, 3.0),
    "green-salad":                      _n(20,  1.0, 4.0,  0.2, 1.5),
    "moong-beans-curry":                _n(90,  5.5, 12.0, 2.5, 4.0),
    "khichdi":                          _n(120, 4.0, 20.0, 2.5, 1.5),
    "lemon-rice":                       _n(155, 3.0, 28.0, 3.5, 1.0),
    "live-roti-with-ghee":              _n(300, 8.5, 48.0, 8.0, 4.0),
    "non-spicy-curry-bottle-gourd":     _n(55,  1.2, 6.0,  3.0, 1.2),
    "papad":                            _n(340, 20.0, 52.0, 2.5, 6.0),
    "plain-rice":                       _n(130, 2.7, 28.0, 0.3, 0.4),
    "watermelon":                       _n(30,  0.6, 7.6,  0.2, 0.4),
    "aloo-fry":                         _n(150, 2.2, 20.0, 7.0, 2.0),
    "banana":                           _n(89,  1.1, 23.0, 0.3, 2.6),
    "mix-fruit":                        _n(50,  0.7, 12.0, 0.2, 1.5),
    "non-spicy-baby-corn-capsicum-dry": _n(70,  2.5, 8.0,  3.0, 2.5),
    "sweet":                            _n(300, 4.0, 50.0, 10.0, 1.0),
    "tomato-rice":                      _n(145, 3.0, 27.0, 3.0, 1.0),
    "fried-papad-rings":                _n(440, 12.0, 50.0, 22.0, 3.0),
    "gravy":                            _n(90,  2.0, 8.0,  5.5, 1.5),
    "ivy-gourd-fry":                    _n(90,  1.5, 8.0,  6.0, 2.5),
    "mango-pickle":                     _n(190, 1.5, 12.0, 15.0, 3.0),
    "papad-chat":                       _n(150, 6.0, 20.0, 5.0, 3.0),
    "pepper-rasam":                     _n(25,  0.9, 3.5,  0.8, 0.5),
    "pineapple":                        _n(50,  0.5, 13.0, 0.1, 1.4),
    "corn-fry":                         _n(130, 3.5, 17.0, 5.5, 2.5),
    "paneer-curry":                     _n(180, 8.0, 6.0,  14.0, 1.0),
    "semiya":                           _n(140, 3.5, 24.0, 3.5, 1.5),
}


def nutrition_for(name, grams):
    """Nutrition dict for `grams` of dish `name`, or None if unknown."""
    per100 = NUTRITION_PER_100G.get(name)
    if per100 is None or grams is None:
        return None
    factor = grams / 100.0
    return {k: round(per100[k] * factor, 1) for k in FIELDS}


def sum_nutrition(items):
    """Sum the 'nutrition' dicts across API items. None if nothing to sum."""
    totals = {k: 0.0 for k in FIELDS}
    counted = False
    for it in items:
        n = it.get("nutrition")
        if n:
            counted = True
            for k in FIELDS:
                totals[k] += n[k]
    return {k: round(v, 1) for k, v in totals.items()} if counted else None


if __name__ == "__main__":
    from food_classes import CLASSES
    missing = [n for i, n in sorted(CLASSES.items()) if i != 0 and n not in NUTRITION_PER_100G]
    print(f"{len(NUTRITION_PER_100G)} classes have nutrition data.")
    if missing:
        print(f"\nMissing ({len(missing)}):")
        for n in missing:
            print("  ", n)
    else:
        print("All classes covered.")