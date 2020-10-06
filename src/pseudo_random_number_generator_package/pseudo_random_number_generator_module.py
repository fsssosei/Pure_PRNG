'''
pseudo_random_number_generator - This is the package used to generate pseudo-random Numbers.
Copyright (C) 2020  sosei
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

import numpy as np

__all__ = ['prng_class']

class prng_class(object):
    '''
        prng_class() -> The system random number as the seed of the pseudo-random number generator.
        
        prng_class(seed) -> A pseudo-random number generator that specifies seeds.
        
        Note
        ----
        The generated instance is thread-safe.
    '''
    
    version = '0.6.0'
    
    prng_period_dict = {'xoshiro256++': 2 ** 256 - 1}
    seed_length_dict = {'xoshiro256++': 256}
    
    def __init__(self, seed: int = None, prng_type: str = 'xoshiro256++'):
        '''
            Set up an instance of a pseudorandom number generator.
            
            Parameters
            ----------
            seed : integer, or None (default)
                Sets the seed for the current instance.
            
            prng_type: str, or 'xoshiro256++' (default)
                Set the pseudo-random number algorithm used for the current instance.
        '''
        self.prng_object_dict = {'xoshiro256++': self.__xoshiro256plusplus}
        
        assert isinstance(seed, (int, type(None))), 'Error: The value of the seed is non-integer.'
        assert prng_type in self.prng_object_dict.keys(), 'Error: The string for prng_type is not in the specified list.'
        
        self.s_array_of_xoshiro256pluspluseed = seed
        self.prng_type = prng_type
        self.__seed_initialization()
    
    def __seed_initialization(self):  #The original seed is hash obfuscated for pseudo-random generation.
        from math import ceil
        from hashlib import blake2b
        
        def seed_length_mask(seed: int) -> int:
            seed &= (1 << self.__class__.seed_length_dict[self.prng_type]) - 1
            return seed
        
        def initializes_seed_for_xoshiro256plusplus():  #Generate the initialization seed for Xoshiro256PlusPlus.
            byte_length_of_seed = ceil(self.__class__.seed_length_dict[self.prng_type] / 8)  #Converts bit length to byte length.
            blake2b_digest_size = byte_length_of_seed
            
            full_zero_bytes_of_digest_size = (0).to_bytes(blake2b_digest_size, byteorder = 'little')
            while True:  #The Xoshiro256PlusPlus algorithm requires that the input seed value is not zero.
                hash_seed_bytes = blake2b(self.s_array_of_xoshiro256pluspluseed.to_bytes(byte_length_of_seed, byteorder = 'little'), digest_size = blake2b_digest_size).digest()
                if hash_seed_bytes == full_zero_bytes_of_digest_size:  #Avoid hash results that are zero.
                    self.s_array_of_xoshiro256pluspluseed += 1  #Changing the seed value produces a new hash result.
                    self.s_array_of_xoshiro256pluspluseed = seed_length_mask(self.s_array_of_xoshiro256pluspluseed)
                else:
                    break
            
            self.s_array_of_xoshiro256plusplus = np.array(np.frombuffer(hash_seed_bytes, dtype = np.uint64))  #Xoshiro256PlusPlus to use the 256 bit uint64 seed array.
        
        seed_initialization_method_dict = {'xoshiro256++': initializes_seed_for_xoshiro256plusplus}
        
        if self.s_array_of_xoshiro256pluspluseed is None:
            from secrets import randbits
            self.s_array_of_xoshiro256pluspluseed = randbits(self.__class__.seed_length_dict[self.prng_type])  #Read unreproducible seeds provided by the operating system.
        else:
            self.s_array_of_xoshiro256pluspluseed = seed_length_mask(self.s_array_of_xoshiro256pluspluseed)

        seed_initialization_method_dict[self.prng_type]()  #The specific initialization seed method is called according to self.prng_type.
        
    @np.errstate(over = 'ignore')
    def __xoshiro256plusplus(self) -> int:  #Xoshiro256PlusPlus algorithm implementation.
        #The external variable used is "self.s_array_of_xoshiro256plusplus".
        
        def rotl(x: np.uint64, k: np.uint64) -> np.uint64:
            return (x << k) | (x >> np.uint64(64 - k))
        
        result = 0
        for i in range(4):
            result |= int(rotl(self.s_array_of_xoshiro256plusplus[0] + self.s_array_of_xoshiro256plusplus[3], np.uint64(23)) + self.s_array_of_xoshiro256plusplus[0]) << (64 * i)
            t = self.s_array_of_xoshiro256plusplus[1] << np.uint64(17)
            self.s_array_of_xoshiro256plusplus[2] ^= self.s_array_of_xoshiro256plusplus[0]
            self.s_array_of_xoshiro256plusplus[3] ^= self.s_array_of_xoshiro256plusplus[1]
            self.s_array_of_xoshiro256plusplus[1] ^= self.s_array_of_xoshiro256plusplus[2]
            self.s_array_of_xoshiro256plusplus[0] ^= self.s_array_of_xoshiro256plusplus[3]
            self.s_array_of_xoshiro256plusplus[2] ^= t
            self.s_array_of_xoshiro256plusplus[3] = rotl(self.s_array_of_xoshiro256plusplus[3], np.uint64(45))
        return result
    
    def source_random_number(self):
        '''
            Straight out of a pseudorandom number generator original random number.
            
            Returns a pseudo-random number.
            
            Note
            ----
            The value type and value range are determined by the algorithm specified by the prng_type parameter when the instance is initialized.
            
            Examples
            --------
            >>> prng_instance = prng_class(170141183460469231731687303715884105727)
            >>> prng_instance.source_random_number()
            73260932800743358445652462028207907455677987852735468159219395093090100006110
        '''
        return self.prng_object_dict[self.prng_type]()
    
    def random_integer_number_with_definite_period(self, new_period: int) -> int:
        '''
            Generates an integer pseudo-random number with a specified period.
            
            Parameters
            ----------
            new_period : integer
                Set the period of the pseudo-random sequence.
            
            Returns a pseudo-random integer for a new period.
            
            Note
            ----
            The value of (new period/original period) is the conversion efficiency. When the difference between the new period and the original period is large, generating a pseudo-random number can be very slow!
            
            Examples
            --------
            >>> prng_instance = prng_class(170141183460469231731687303715884105727)
            >>> prng_instance.random_integer_number_with_definite_period(115792089237316195423570985008687907853269984665640564039457584007913129639747)
            73260932800743358445652462028207907455677987852735468159219395093090100006110
        '''
        assert isinstance(new_period, int), 'Error: The value of new_period is non-integer.'
        assert new_period <= self.__class__.prng_period_dict[self.prng_type], 'Error: Suppose the new period number cannot be greater than the original period number of the pseudorandom number generator.'
        
        random_number = self.s_array_of_xoshiro256plusplusource_random_number()
        assert isinstance(random_number, int), 'Error: The chosen pseudo-random number generator is non-integer.'
        while True:
            if random_number < new_period:
                return random_number
            random_number = self.s_array_of_xoshiro256plusplusource_random_number()
