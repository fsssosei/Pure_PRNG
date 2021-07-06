'''
ICG - This package is the inversive congruential pseudo-random generator.
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
from gmpy2 import mpz, invert as gmpy2_invert
from rng_util_package import bit_length_mask

__all__ = ['ICG']

Integer = TypeVar('Integer', int, mpz)

class ICG:
    '''
        Inversive congruential generators are a type of nonlinear congruential pseudorandom number generator.
        The period of this random generator is 2^256*102
        
        References
        ----------
        "Inversive congruential generator"
        https://en.wikipedia.org/wiki/Inversive_congruential_generator
    '''
    
    version = '1.0.0'
    
    def __icg(self) -> Integer:
        q = self.q
        a=mpz(0x2042043bbb1713c5bbb2692534d0cd8c9135e42b762b1a7d35f70836aa50d9c37)
        c=mpz(0x52cffb72243ddbd01b6d67c723eec63b60aec9cd7f8916b1943bf1a15a6e98cc94)
        
        if self.x == 0:
            self.x = c
        else:
            self.x = (a * gmpy2_invert(self.x, q) + c) % q
        
        return self.x
    
    
    def __init__(self, seed: Integer):
        '''
            Create an instance of a pseudo-random number generator.  创建一个伪随机数生成器的实例。
            
            Parameters
            ----------
            seed: Integer
                Sets the seed for the current instance.
                Any number whose seed is greater than 2^256*102 will be modulo 2^256*102+1
        '''
        self.q = mpz(2 ** 256 * 102 + 1)
        self.x = mpz(seed % self.q)
    
    
    def random_raw(self) -> Integer:
        x = self.__icg()
        if (x >> 256) == 102:
            x = self.__icg()
        return bit_length_mask(x, 256)
