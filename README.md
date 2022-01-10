# MIX python

This package is both a tutorial paper and a library of classic TAOCP algorithms in the Python language

_**The Art of Computer Programming ([TAOCP](https://en.wikipedia.org/wiki/The_Art_of_Computer_Programming))**_ is a comprehensive monograph written by computer scientist Donald Knuth that covers many kinds of programming algorithms and their analysis.

Install:

`pip install mixpython`

Simple usage:

```
import mixpython.algorithms as mix

alg = mix.Algorithms(logger=True) # mix.Algorithms - A base class that implements the algorithms and minimum settings

alg.euclid_e(45678, 86)
```