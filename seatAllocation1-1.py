# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:38:53 2015

@author: haslam
"""
from math import sqrt
import sys
sys.setrecursionlimit(50001)

class Memoize:
    "Via: http://stackoverflow.com/a/1988826/553404"
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

def seats(n):
    """
    Returns expected number of seats for process descibed as
    """
    if n == 1:
        return 1
    elif n <= 0:
        return 0
    else:
        s = 0.0
        for k in range(n):
            s +=  1.0 + seats(n - k - 2) + seats(k - 1)
        return s / n


def seat_variance(n):
     """
     Returns variance of number of seats for process...
     """
     if n ==3:
         var = ((1 - 5/3.0)**2 + (2 - 5/3)**2 + (2 - 5/3)**2)/3.0 #Expected Value is 5/3 for n=3
         return var
     elif n <= 1:
         return 0.0
     else:
         v = 0.0
         for k in range(n):
             v += seat_variance(n - k - 2) + seat_variance(k - 1)
     return v / n        


seats = Memoize(seats)
seat_variance = Memoize(seat_variance)

N = 25
print(N, "seats, expected occupancy", seats(N)/N)
print(N, "seats, var", seat_variance(N), "std", sqrt(seat_variance(N)))