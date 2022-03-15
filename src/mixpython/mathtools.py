from logger import logger


def floor_div(a: int, b: int, logger: bool = logger):
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
    def __init__(self, logger: bool = logger):
        self.logger = logger

    def base_proof(self):
        pass

