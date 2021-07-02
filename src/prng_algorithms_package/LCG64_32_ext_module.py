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

from ctypes import c_uint32, c_uint64
from array import array
from gmpy2 import bit_mask as gmpy2_bit_mask
from rng_util_package import rotl, rotr

__all__ = ['LCG64_32_ext']

class LCG64_32_ext:
    '''
        The period of this random generator is 2^(32*(2^n+2)).
        
        References
        ----------
        "伪随机数生成算法"
        https://baobaobear.github.io/post/20200104-xoshiro/
    '''
    
    version = '1.0.0'
    
    def __init__(self, seed: int, n: int = 1):
        self.n = n
        self.ext_size = 1 << n
        
        self.mask32 = gmpy2_bit_mask(32)
        ext_seed_multiplier = c_uint64(6364136223846793005).value
        ext_seed_addend = c_uint64(1).value
        
        init_num_of_iter = 256 * 256
        
        seed = c_uint64(seed).value
        ext_seed = c_uint64(seed * ext_seed_multiplier + ext_seed_addend).value
        
        self.s = c_uint64(seed).value
        self.a_array = array('L')
        self.a_array.append(seed >> 32)
        if self.ext_size >= 2:
            self.a_array.append(ext_seed & self.mask32)
            if self.ext_size >= 3:
                self.a_array.append(ext_seed >> 32)
                for _ in range(3, self.ext_size):
                    self.a_array.append(rotl((self.a_array[-1] * ext_seed_multiplier + ext_seed_addend) & self.mask32, 32, self.a_array[-2] & 0xF) ^ self.a_array[-2])
        
        for i in range(init_num_of_iter):
            self.random_raw()
    
    def random_raw(self):
        a_array_multiplier = c_uint32(2891336453).value
        a_array_addend = c_uint32(887987685).value
        s_multiplier = c_uint64(3935559000370003845).value
        s_addend = c_uint64(1442695040888963407).value
        if self.s == 0:
            carry = False
            for i in range(self.ext_size):
                if carry:
                    self.a_array[i] = (self.a_array[i] * a_array_multiplier + a_array_addend) & self.mask32
                    carry = (self.a_array[i] == 0)
                self.a_array[i] = (self.a_array[i] * a_array_multiplier + a_array_addend) & self.mask32
                carry |= (self.a_array[i] == 0)
        prev_s = self.s
        self.s = (self.s * s_multiplier + s_addend) & gmpy2_bit_mask(64)
        return rotr(self.s >> 32, 32, prev_s >> 59) ^ self.a_array[prev_s & gmpy2_bit_mask(self.n)]