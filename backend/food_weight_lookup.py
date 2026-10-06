"""
food_weight_lookup.py
----------------------
Per-class physical properties used to turn (area, height) into a weight
estimate. Rough, common-knowledge approximations, NOT measured for this
dataset. Tune against a few real weighed plates.

"volume" type:  weight_g = area_cm2 * height_cm * density_g_cm3
"flat" type:    weight_g = typical_weight_g * (area_cm2 / typical_area_cm2)

Keys match food_classes.py exactly. Run `python food_weight_lookup.py`
to list any class that is missing.
"""


def _vol(d):
    return {"type": "volume", "density_g_cm3": d}


FOOD_PROPERTIES = {
    "aloo-dry-fry":                     _vol(0.75),
    "avakaya-muddha-pappu-rice":        _vol(0.95),
    "baby-corn-capsicum-dry":           _vol(0.60),
    "cabbage-pakodi":                   _vol(0.45),
    "cabbage-fry":                      _vol(0.55),
    "capsicum-paneer-curry":            _vol(0.95),
    "chakar-pongal":                    _vol(1.00),
    "chole-masala":                     _vol(0.95),
    "cluster-beans-curry":              _vol(0.75),
    "cucumber-raita":                   _vol(1.02),
    "gobi-masala-curry":                _vol(0.80),
    "gutti-vankaya-curry":              _vol(0.90),
    "jeera-rice":                       _vol(0.90),
    "mixed-curry":                      _vol(0.90),
    "muskmelon":                        _vol(0.90),
    "rajma":                            _vol(0.95),
    "rasgulla":                         _vol(1.10),
    "sambar":                           _vol(1.00),
    "tomato-rasam":                     _vol(1.00),
    "vankaya-ali-karam":                _vol(0.80),
    "veg-biriyani":                     _vol(0.85),
    "aloo-curry":                       _vol(0.95),
    "curd":                             _vol(1.03),
    "dal":                              _vol(1.00),
    "fresh-chutney":                    _vol(1.05),
    "green-salad":                      _vol(0.50),
    "moong-beans-curry":                _vol(0.95),
    "khichdi":                          _vol(1.00),
    "lemon-rice":                       _vol(0.90),
    "live-roti-with-ghee":              {"type": "flat", "typical_weight_g": 38, "typical_area_cm2": 180},
    "non-spicy-curry-bottle-gourd":     _vol(0.90),
    "papad":                            {"type": "flat", "typical_weight_g": 10, "typical_area_cm2": 95},
    "plain-rice":                       _vol(0.90),
    "watermelon":                       _vol(0.95),
    "aloo-fry":                         _vol(0.75),
    "banana":                           _vol(0.95),
    "mix-fruit":                        _vol(0.90),
    "non-spicy-baby-corn-capsicum-dry": _vol(0.60),
    "sweet":                            _vol(0.95),
    "tomato-rice":                      _vol(0.90),
    "fried-papad-rings":                _vol(0.35),
    "gravy":                            _vol(1.00),
    "ivy-gourd-fry":                    _vol(0.65),
    "mango-pickle":                     _vol(1.00),
    "papad-chat":                       _vol(0.50),
    "pepper-rasam":                     _vol(1.00),
    "pineapple":                        _vol(0.90),
    "corn-fry":                         _vol(0.70),
    "paneer-curry":                     _vol(0.95),
    "semiya":                           _vol(0.80),
}


if __name__ == "__main__":
    from food_classes import CLASSES
    names = {n for i, n in CLASSES.items() if i != 0}
    missing = sorted(names - set(FOOD_PROPERTIES))
    extra = sorted(set(FOOD_PROPERTIES) - names)
    print(f"{len(FOOD_PROPERTIES)} entries.")
    print("Missing:", missing or "none")
    print("Not in food_classes.py:", extra or "none")