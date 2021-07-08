'''
LCG64_32_ext - This package is the LCG64_32_ext pseudo-random generator.
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
from ctypes import c_uint32, c_uint64
from array import array
from gmpy2 import mpz
from rng_util_package import bit_length_mask, rotl, rotr

__all__ = ['LCG64_32_ext']

Integer = TypeVar('Integer', int, mpz)

class LCG64_32_ext:
    '''
        This is an extended linear congruence generator algorithm.
        
        References
        ----------
        "伪随机数生成算法"
        https://baobaobear.github.io/post/20200104-xoshiro/
    '''
    
    version = '1.1.3'
    
    def __init__(self, seed: Integer, n: Integer = 1):
        '''
            Create an instance of a pseudo-random number generator.  创建一个伪随机数生成器的实例。
            
            Parameters
            ----------
            seed: Integer
                Sets the seed for the current instance.
                Any input seed that is longer than 64 bits will be truncated to the lower 64 bits.
            
            n: Integer, default integer 1
                Set the parameters for the period.
                The period of this random generator is 2^(32*(2^n+2))
        '''
        self.n = n
        self.ext_size = 1 << n
        
        ext_seed_multiplier = c_uint64(6364136223846793005).value
        ext_seed_addend = c_uint64(1).value
        
        init_num_of_iter = 2 ** 16
        
        seed = c_uint64(seed).value
        ext_seed = c_uint64(seed * ext_seed_multiplier + ext_seed_addend).value
        
        self.s = c_uint64(seed).value
        self.a_array = array('L')
        self.a_array.append(seed >> 32)
        if self.ext_size >= 2:
            self.a_array.append(bit_length_mask(ext_seed, 32))
            if self.ext_size >= 3:
                self.a_array.append(ext_seed >> 32)
                for _ in range(3, self.ext_size):
                    self.a_array.append(rotl(bit_length_mask(self.a_array[-1] * ext_seed_multiplier + ext_seed_addend, 32), 32, self.a_array[-2] & 0xF) ^ self.a_array[-2])
        
        for i in range(init_num_of_iter):
            self.random_raw()
    
    
    def random_raw(self) -> Integer:
        a_array_multiplier = c_uint32(2891336453).value
        a_array_addend = c_uint32(887987685).value
        s_multiplier = c_uint64(3935559000370003845).value
        s_addend = c_uint64(1442695040888963407).value
        s = self.s
        a_array = self.a_array
        if s == 0:
            carry = False
            for i in range(self.ext_size):
                if carry:
                    a_array[i] = bit_length_mask(a_array[i] * a_array_multiplier + a_array_addend, 32)
                    carry = (a_array[i] == 0)
                a_array[i] = bit_length_mask(a_array[i] * a_array_multiplier + a_array_addend, 32)
                carry |= (a_array[i] == 0)
        prev_s = s
        s = bit_length_mask(s * s_multiplier + s_addend, 64)
        self.s = s
        return rotr(s >> 32, 32, prev_s >> 59) ^ a_array[bit_length_mask(prev_s, self.n)]
