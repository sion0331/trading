def is_pos(x):  # treat None/neg/zero as not positive
    try:
        return x is not None and float(x) > 0
    except:
        return False


def is_equal(a, b, eps=1e-12):
    # consider NaN/None differences as change
    try:
        return abs(float(a) - float(b)) < eps
    except:
        return False
