'''
XSM64 - This package is the XSM64 pseudo-random generator.
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
from rng_util_package import bit_length_mask, rotl

__all__ = ['XSM64']

Integer = TypeVar('Integer', int, mpz)

class XSM64:
    '''
        This is a PRNG that combines a 128-bit LCG state transition function with a complex nonlinear output function.
        The period of this random generator is 2^128
        
        References
        ----------
        "pracrand"
        http://pracrand.sourceforge.net/RNG_engines.txt
    '''
    
    version = '1.0.3'
    
    def __step_forwards(self):
        tmp = bit_length_mask(self.lcg_low + self.lcg_adder_high, 64)
        self.lcg_low = bit_length_mask(self.lcg_low + self.lcg_adder_low, 64)
        self.lcg_high = bit_length_mask(self.lcg_high + tmp + (1 if self.lcg_low < self.lcg_adder_low else 0), 64)
    
    
    def __init__(self, seed: Integer):
        '''
            Create an instance of a pseudo-random number generator.  创建一个伪随机数生成器的实例。
            
            Parameters
            ----------
            seed: Integer
                Sets the seed for the current instance.
                Any input seed that is longer than 64 bits will be truncated to the lower 64 bits.
        '''
        self.lcg_adder_low = c_uint64(1).value
        self.lcg_adder_high = c_uint64(seed << 1).value
        
        self.lcg_low = self.lcg_adder_low
        self.lcg_high = self.lcg_adder_high ^ ((bit_length_mask(seed, 64) >> 63) << 63)

        self.__step_forwards()
    
    
    def random_raw(self) -> Integer:
        K = c_uint64(0xA3EC647659359ACD).value
        
        lcg_high = self.lcg_high
        
        tmp = lcg_high ^ rotl(bit_length_mask(lcg_high + self.lcg_low, 64), 64, 16)
        tmp ^= rotl(bit_length_mask(tmp + self.lcg_adder_high, 64), 64, 40)
        tmp = bit_length_mask(tmp * K, 64)
        
        self.__step_forwards()
        
        tmp ^= rotl(bit_length_mask(tmp + lcg_high, 64), 64, 32)
        tmp = bit_length_mask(tmp * K, 64)
        tmp ^= tmp >> 32
        
        return tmp
