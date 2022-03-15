from logger import logger
from mathtools import floor_div

"""A base class that implements the algorithms and minimum settings"""


def e(a: int, b: int):
    """
    Euclidean algorithm(GCD) - An effective algorithm for finding the greatest common divisor of two integers

    :param a: int first value
    :param b: int second value
    :return: int GCD value
    """
    while True:

        r = floor_div(a, b)[1]

        if r == 0:
            break

        a = b
        b = r

    return b


def ef(a: int, b: int):
    """
    Euclidean algorithm(GCD)*different implementation - An effective algorithm for finding
    the greatest common divisor of two integers

    :param a: int first value
    :param b: int second value
    :return: int GCD value
    """
    lst = [a, b]
    lst.sort(reverse=True)
    a, b = lst

    while True:
        a = floor_div(a, b)[1]
        if a == 0:
            if logger:
                print(f'result: {b}')
            return b

        b = floor_div(b, a)[1]
        if b == 0:
            if logger:
                print(f'result: {a}')
            return a


def eo(m, n):
    var_list = []
    b = 1
    a_ = b
    b_ = 0
    a = b_
    c, d = m, n

    if logger:
        print(f'eo.1: {a_, a, b_, b, c, d}')

    while True:
        q, r = floor_div(c, d)
        var_list.append([a_, a, b_, b, c, d, q, r])

        if logger:
            print(f'eo.2: {q, r} = {c} / {d}')
            print(f'eo.3: r = {r}')

        if r == 0:
            break

        c = d
        d = r
        t = a_
        a_ = a
        a = t - q * a
        t = b_
        b_ = b
        b = t - q * b

        if logger:
            print(f'eo.4: {var_list}')

    if logger:
        print(f'eo.end: {a} * {m} + {b} * {n} = {a * m} + {b * n} = {a * m + b * n} = {d}')
        print(f'eo.end: {a, b} {d}')
    return a, b, d, var_list
