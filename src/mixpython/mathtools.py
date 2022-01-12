from logger import logger as log

LOGGER = log


def floor_div(a: int, b: int, logger: bool = LOGGER):
    """
    Integer division, returns the quotient and remainder

    :param logger:
    :param a: int  dividend
    :param b: int divider
    :return: tuple (div, mod)
    """
    d, m = a // b, a % b
    if logger:
        print(f'{a}/{b} div: {d}, mod: {m}')
    return d, m


class Induction:
    def __init__(self, logger: bool = LOGGER):
        self.logger = logger

    def base_proof(self):
        pass

