'''
Ran64 - This package is the Ran64 pseudo-random generator.
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
from rng_util_package import bit_length_mask

__all__ = ['Ran64']

Integer = TypeVar('Integer', int, mpz)

class Ran64:
    '''
        A kind of LCG plus mixed PRNG.
        The period of this random generator is 2^64
        
        References
        ----------
        "Random Numbers"
        https://people.sissa.it/~inno/pubs/rng-2018.pdf
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
        self.v = c_uint64(0x38ecac5fb3251641).value
        self.w = c_uint64(1).value
        self.u = bit_length_mask(seed, 64) ^ self.v
        self.random_raw()
        self.v = self.u
        self.random_raw()
        self.w = self.v
        self.random_raw()
    
    
    def random_raw(self) -> Integer:
        u = self.u
        u = bit_length_mask(u * 0x27bb2ee687b0b0fd + 0x9c740a0a6788d2c, 64)
        self.u = u
        
        v = self.v
        v ^= v >> 17
        v &= bit_length_mask(v << 31, 64)
        v ^= v >> 8
        self.v = v
        
        w = self.w
        w = bit_length_mask(w * 0xffffda61 + (w >> 32), 32)
        self.w = w
        
        x = u ^ bit_length_mask(u << 21, 64)
        x ^= x >> 35
        x ^= bit_length_mask(x << 4, 64)
        return bit_length_mask(x + v, 64) ^ w
