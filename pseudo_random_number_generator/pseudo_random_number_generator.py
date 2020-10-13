'''
pseudo_random_number_generator - This package is used to generate multi-precision pseudo-random Numbers.
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

from typing import Optional, Union
import gmpy2
import numpy as np

__all__ = ['prng_class']

class prng_class(object):
    '''
        Generate multi-precision pseudo-random Numbers.
        There are "methods" that specify the period of a pseudo-random sequence.
        
        prng_class() -> The system random number as the seed of the pseudo-random number generator.
        
        prng_class(seed) -> A pseudo-random number generator that specifies seeds.
        
        prng_class(prng_type = 'xoshiro256++') -> Set the pseudo-random number generator algorithm used.
        
        Note
        ----
        The generated instance is thread-safe.
        Only the pseudo-random number generation algorithm with period of 2^n or 2^n-1 is adapted.
        The pseudo-random number generation algorithm implemented here must be the full-period length output.
    '''
    
    VERSION = '0.7.2'
    
    prng_period_dict = {'xoshiro256++': 2 ** 256 - 1}
    seed_length_dict = {'xoshiro256++': 256}
    
    def __init__(self, seed: Optional[int] = None, prng_type: str = 'xoshiro256++') -> None:
        '''
            Set up an instance of a pseudorandom number generator.
            
            Parameters
            ----------
            seed : integer, or None (default)
                Sets the random seed.
            
            prng_type: str, or 'xoshiro256++' (default)
                Set the pseudo-random number algorithm.
        '''
        self.algorithms_dict = {'xoshiro256++': self.__xoshiro256plusplus}
        
        assert isinstance(seed, (int, type(None))), 'Error: The value of the seed is non-integer.'
        if isinstance(seed, int): assert seed >= 0, "Error: seed can't be negative."
        assert prng_type in self.algorithms_dict.keys(), 'Error: The string for prng_type is not in the specified list.'
        
        self.seed = self.__seed_initialization(seed, prng_type)
        self.prng_type = prng_type
    
    def __seed_initialization(self, seed: int, prng_type: str) -> None:  #The original seed is hash obfuscated for pseudo-random generation.
        from math import ceil
        from hashlib import blake2b
        
        def seed_length_mask(seed: int, prng_type: str) -> int:
            seed &= (1 << self.__class__.seed_length_dict[prng_type]) - 1
            return seed
        
        def seed_initialize_xoshiro256plusplus(seed: int, prng_type: str) -> None:  #Generate the self variable used by the Xoshiro256PlusPlus algorithm.
            seed_byte_length = ceil(self.__class__.seed_length_dict[prng_type] / 8)  #Converts bit length to byte length.
            blake2b_digest_size = seed_byte_length
            
            full_zero_bytes = (0).to_bytes(blake2b_digest_size, byteorder = 'little')
            while True:  #The Xoshiro256PlusPlus algorithm requires that the input seed value is not zero.
                hash_seed_bytes = blake2b(seed.to_bytes(seed_byte_length, byteorder = 'little'), digest_size = blake2b_digest_size).digest()
                if hash_seed_bytes == full_zero_bytes:  #Avoid hash results that are zero.
                    seed += 1  #Changing the seed value to produce a new hash result.
                    seed = seed_length_mask(seed, prng_type)
                else:
                    break
            
            self.s_array_of_xoshiro256plusplus = np.array(np.frombuffer(hash_seed_bytes, dtype = np.uint64))  #Xoshiro256PlusPlus to use the 256 bit uint64 seed array.
        
        seed_init_algorithm_dict = {'xoshiro256++': seed_initialize_xoshiro256plusplus}
        
        if seed is None:
            from secrets import randbits
            seed = randbits(self.__class__.seed_length_dict[prng_type])  #Read unreproducible seeds provided by the operating system.
        else:
            seed = seed_length_mask(seed, prng_type)

        seed_init_algorithm_dict[prng_type](seed, prng_type)  #The specific initialization seed method is called according to prng_type.
        return seed
        
    @np.errstate(over = 'ignore')
    def __xoshiro256plusplus(self) -> int:  #Xoshiro256PlusPlus method realizes full-period length output, [1, 2^ 256-1]
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
    
    def source_random_number(self) -> Union[int, float]:
        '''
            The source random number directly derived from the random generator algorithm.
            The result can be processed into other data types in other methods.

            Returns a pseudo-random number.
            
            Note
            ----
            The value type and value range are determined by the random algorithm,
            which is specified by parameter `prng_type` at instance initialization.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = prng_class(seed)
            >>> prng_instance.source_random_number()
            73260932800743358445652462028207907455677987852735468159219395093090100006110
        '''
        return self.algorithms_dict[self.prng_type]()
    
    def rand_float(self, period: Optional[int] = None) -> gmpy2.mpfr:
        '''
            Parameters
            ----------
            period : integer
                Set the period of the pseudo-random sequence.

            Returns a random float in [0, 1], with 0 included and 1 excluded.
            The period will be set to a rational value if not specified.
            The return type is `gmpy2.mpfr`.
            Returns a real pseudo-random number of type gmpy2.mpfr of the adaptive period length of a range of [0, 1].
            0 (inclusive) and 1 (exclusive).

            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = prng_class(seed)
            >>> prng_instance.rand_float()
            mpfr('0.6326937641706669741872583730940429737405414921354622618051716414693676562568173',257)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.rand_float(period)
            mpfr('0.02795744845257346733436109648463446736744766610965612207643215290679786849301309',257)
        '''
        prng_period = self.__class__.prng_period_dict[self.prng_type]
        prng_period_bit_size = prng_period.bit_length() + ((prng_period & 1) - 1)
        with gmpy2.local_context(gmpy2.context(), precision = prng_period_bit_size + 1):
            if period is None:
                return gmpy2.mpfr(self.source_random_number()) / gmpy2.mpfr(prng_period)
            else:
                return gmpy2.mpfr(self.randint_with_period(period)) / gmpy2.mpfr(prng_period)
    
    def randint(self, b: int, a: int = 0, period: Optional[int] = None) -> int:
        '''
            Parameters
            ----------
            b : integer
                Upper bound on the range including `b`.
                
            a : integer, or 0 (default)
                Lower bound on the range including `a`.
            
            period : integer
                Set the period of the pseudo-random sequence.
            
            Returns an integer pseudo-random number in the range [a, b].
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = prng_class(seed)
            >>> prng_instance.randint(100, 1)
            64
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.randint(100, 1, period)
            3
        '''
        assert isinstance(b, int), 'Error: The value of b is non-integer.'
        assert isinstance(a, int), 'Error: The value of a is non-integer.'
        
        scale = b - a + 1
        return a + int(scale * self.rand_float(period))
    
    def randint_set(self, b: int, a: int, k: int) -> set:
        '''
            Parameters
            ----------
            b : integer
                Upper bound on the range including `b`.
                
            a : integer
                Lower bound on the range including `a`.
                
            k : integer
                The number of set elements to generate.
            
            Returns a set of pseudo-random Numbers with k elements in the range [a, b].
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = prng_class(seed)
            >>> prng_instance.randint_set(100, 1, 6)
            {64, 39, 9, 41, 23, 92}
        '''
        assert isinstance(a, int), 'Error: The value of a is non-integer.'
        assert isinstance(b, int), 'Error: The value of b is non-integer.'
        assert isinstance(k, int), 'Error: The value of k is non-integer.'
        assert a <= b, 'Error: a cannot be greater than b.'
        assert k >= 0, "Error: k can't be negative."
        assert k <= (b - a), "Error: k can't be greater than b minus a."
        
        result = set()
        while len(result) < k:
            result.add(self.randint(b, a))
        return result
    
    def randint_with_period(self, period: int) -> int:
        '''
            Generates an integer pseudo-random number with a specified period.
            
            Parameters
            ----------
            period : integer
                Set the period of the pseudo-random sequence.
            
            Returns a pseudo-random integer for a new period.
            
            Note
            ----
            The value of (new period / original period) is the representation of generating efficiency.
            When the difference between the new period and the original period is too large,
            it may takes a **long** time to generate a pseudo-random number!
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = prng_class(seed)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.randint_with_period(period)
            40688839126177430252467309162469901643963863918059158449302074429100738061375
        '''
        assert isinstance(period, int), 'Error: The value of new_period is non-integer.'
        prng_period = self.__class__.prng_period_dict[self.prng_type]
        assert period > 0, 'Error: new_period must be greater than zero.'
        assert period <= prng_period, 'Error: Suppose the new period number cannot be greater than the original period number of the pseudorandom number generator.'
        
        number_to_subtract = prng_period - period
        set_of_numbers_to_exclude = self.randint_set(prng_period, prng_period & 1, number_to_subtract)
        
        random_number = self.source_random_number()
        assert isinstance(random_number, int), 'Error: The chosen pseudo-random number generator is non-integer.'
        while True:
            if random_number not in set_of_numbers_to_exclude:
                return random_number
            random_number = self.source_random_number()
