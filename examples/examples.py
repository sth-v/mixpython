import mixpython.algorithms as mix

# mix.Algorithms - A base class that implements the algorithms and minimum settings
alg = mix.Algorithms(logger=True)

'''The 'logger' variable includes the implementation of a simple logging system for the algorithms to make their work 
clear. It is recommended to use the value of the variable logger = 'True' while learning. When using the algorithms 
in your own applications use the default value or 'False'.
'''

alg.euclid_e(45678, 86)
alg.euclid_f(45678, 86)