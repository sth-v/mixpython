from logger import logger as log
from mathtools import floor_div
LOGGER = log


class Algorithms:
    """A base class that implements the algorithms and minimum settings
    """
    def __init__(self, logger: bool = True):
        self.logger = logger
        '''The 'logger' variable includes the implementation of a simple logging system for the algorithms to make their 
        work clear. It is recommended to use the value of the variable logger = 'True' while learning. 
        When using the algorithms in your own applications use the default value or 'False'.
        '''

    @staticmethod
    def pack(*args):
        """
        service function for packing a motorcade into a list
        :param args: tuple of args
        :return: list of args
        """

        return [*args]

    @staticmethod
    def sort(args: list, key: None = None, reverse: bool = True):
        """
        Sort the list in ascending order and return list

        The sort is in-place (i.e. the list itself is modified) and stable (i.e. the order of two equal elements is
        maintained).
        If a key function is given, apply it once to each list item and sort them, ascending or descending, according
        to their function values.

        The reverse flag can be set to sort in descending order.

        :param args: list of args
        :param key: key function
        :param reverse: flag for sort in descending order.
        :return: sorted list
        """
        args.sort(key=key, reverse=reverse)
        return args

    def e(self, a: int, b: int):
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

        if self.logger:
            print(f'result: {b}')
        return b

    def ef(self, a: int, b: int):
        """
        Euclidean algorithm(GCD)*different implementation - An effective algorithm for finding
        the greatest common divisor of two integers

        :param a: int first value
        :param b: int second value
        :return: int GCD value
        """
        a, b = self.sort(self.pack(a, b), reverse=True)

        while True:
            a = floor_div(a, b)[1]
            if a == 0:
                if self.logger:
                    print(f'result: {b}')
                return b

            b = floor_div(b, a)[1]
            if b == 0:
                if self.logger:
                    print(f'result: {a}')
                return a

    class eo:
        def __init__(self, m: int, n: int, logger: bool = True):
            self.m = m
            self.n = n
            self.logger = logger
            self.a = self._eo()[0]
            self.b = self._eo()[1]
            self.gsd = self._eo()[2]
            self.arr = self._eo()[3]

        def _eo(self):
            m = self.m
            n = self.n
            var_list = []
            b = 1
            a_ = b
            b_ = 0
            a = b_
            c , d= m, n

            if self.logger:
                print(f'eo.1: {a_, a, b_, b, c, d}')

            while True:
                q, r = floor_div(c, d)
                var_list.append([a_, a, b_, b, c, d, q, r])

                if self.logger:
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


                if self.logger:
                    print(f'eo.4: {var_list}')


            if self.logger:

                print(f'eo.end: {a} * {m} + {b} * {n} = {a*m} + {b*n} = {a*m + b*n} = {d}')
                print(f'eo.end: {a, b} {d}')
            return a, b, d, var_list

