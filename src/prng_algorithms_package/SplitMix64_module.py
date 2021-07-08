'''
SplitMix64 - This package is the SplitMix64 pseudo-random generator.
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
from gmpy2 import mpz
from rng_util_package import bit_length_mask

__all__ = ['SplitMix64']

Integer = TypeVar('Integer', int, mpz)

class SplitMix64:
    '''
        A variant of counter-based PRNG.
        The period of this random generator is 2^64
        
        References
        ----------
        "SplitMix generator family."
        http://docs.random.dlang.io/latest/mir_random_engine_splitmix.html
    '''
    
    version = '1.0.1'
    
    def __init__(self, seed: Integer):
        '''
            Create an instance of a pseudo-random number generator.  创建一个伪随机数生成器的实例。
            
            Parameters
            ----------
            seed: Integer
                Sets the seed for the current instance.
                Any input seed that is longer than 64 bits will be truncated to the lower 64 bits.
        '''
        self.state = bit_length_mask(seed, 64)
    
    
    def random_raw(self) -> Integer:
        self.state = bit_length_mask(self.state + 0x9e3779b97f4a7c15, 64)
        z = self.state
        z = bit_length_mask((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9, 64)
        z = bit_length_mask((z ^ (z >> 27)) * 0x94d049bb133111eb, 64)
        return z ^ (z >> 31)
