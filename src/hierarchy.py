# Label name mappings
L1_LABEL_NAMES = {0: "Hard News", 1: "Soft News"}
L2_LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# L2 label -> parent L1 label
L2_TO_L1 = {
    0: 0,   # World    -> Hard News
    2: 0,   # Business -> Hard News
    1: 1,   # Sports   -> Soft News
    3: 1,   # Sci/Tech -> Soft News
}

# L1 label -> valid child L2 labels
L1_TO_VALID_L2 = {
    0: [0, 2],   # Hard News -> World, Business
    1: [1, 3],   # Soft News -> Sports, Sci/Tech
}


def is_consistent(l1_pred, l2_pred):
    """Return True if the L1 and L2 predictions are parent-child compatible."""
    return l2_pred in L1_TO_VALID_L2.get(l1_pred, [])
