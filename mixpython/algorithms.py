import collections
import itertools
import pydoc

LOGGER = True


def my_mod(a:int, b:int, logger=LOGGER):
    d, m = a//b, a%b
    if logger:
        print(f'div: {d}, mod: {m}')
    return(d, m)


def euclid(a:int, b:int, logger=LOGGER):

    while True:
         
        r = my_mod(a, b)[1]
        
        if r ==0:
            break
        
        a = b
        b = r
        
    if logger:
        print(f'result:{b}')    
    return b
       


