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

    def my_mod(self, a: int, b: int):
        """
        Integer division, returns the quotient and remainder

        :param a: int  dividend
        :param b: int divider
        :return: tuple (div, mod)
        """
        d, m = a // b, a % b
        if self.logger:
            print(f'{a}/{b} div: {d}, mod: {m}')
        return d, m

    def euclid_e(self, a: int, b: int):
        """
        Euclidean algorithm(GCD) - An effective algorithm for finding the greatest common divisor of two integers

        :param a: int first value
        :param b: int second value
        :return: int GCD value
        """
        while True:

            r = self.my_mod(a, b)[1]

            if r == 0:
                break

            a = b
            b = r

        if self.logger:
            print(f'result: {b}')
        return b

    def euclid_f(self, a: int, b: int):
        """
        Euclidean algorithm(GCD)*different implementation - An effective algorithm for finding
        the greatest common divisor of two integers

        :param a: int first value
        :param b: int second value
        :return: int GCD value
        """
        a, b = self.sort(self.pack(a, b), reverse=True)

        while True:
            a = self.my_mod(a, b)[1]
            if a == 0:
                if self.logger:
                    print(f'result: {b}')
                return b

            b = self.my_mod(b, a)[1]
            if b == 0:
                if self.logger:
                    print(f'result: {a}')
                return a
