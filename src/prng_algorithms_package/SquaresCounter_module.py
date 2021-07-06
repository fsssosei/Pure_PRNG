'''
SquaresCounter - This package is the SquaresCounter pseudo-random generator.
Copyright (C) 2021  sosei

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import TypeVar
from ctypes import c_uint64
from gmpy2 import mpz
from rng_util_package import bit_length_mask, rotr

__all__ = ['SquaresCounter']

Integer = TypeVar('Integer', int, mpz)

class SquaresCounter:
    '''
        A new counter-based RNG based on Johnvon Neumann’s middle square. 
        The period of this random generator is 2^64
        
        References
        ----------
        "Squares: A Fast Counter-Based RNG"
        https://arxiv.org/pdf/2004.06278v3.pdf
    '''
    
    version = '1.0.0'
    
    def __init__(self, seed: Integer):
        '''
            Create an instance of a pseudo-random number generator.  创建一个伪随机数生成器的实例。
            
            Parameters
            ----------
            seed: Integer
                Sets the seed for the current instance.
                Any input seed that is longer than 64 bits will be truncated to the lower 64 bits.
        '''
        self.counter = bit_length_mask(seed, 64)
    
    
    def random_raw(self) -> Integer:
        KEY = c_uint64(0xdea8c96f2e836c30).value  #The magic number. The KEY should be an irregular bit pattern with roughly half ones andhalf zeros.
        
        self.counter = bit_length_mask(self.counter + 1, 64)
        
        y = x = bit_length_mask(self.counter * KEY, 64)
        z = bit_length_mask(y + KEY, 64)
        x = bit_length_mask(x ** 2 + y, 64); x = rotr(x, 64, 32)
        x = bit_length_mask(x ** 2 + z, 64); x = rotr(x, 64, 32)
        x = bit_length_mask(x ** 2 + y, 64); x = rotr(x, 64, 32)
        return bit_length_mask(x ** 2 + z, 64) >> 32
