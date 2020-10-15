'''
pure_prng - This package is used to generate multi-precision pseudo-random Numbers.
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

from typing import Final, Optional, Union, Set
import gmpy2
import numpy as np

__all__ = ['pure_prng']

class pure_prng(object):
    '''
        Generate multi-precision pseudo-random Numbers.
        There are "methods" that specify the period of a pseudo-random sequence.
        
        pure_prng() -> The system random number as the seed of the pseudo-random number generator.
        
        pure_prng(seed) -> A pseudo-random number generator that specifies seeds.
        
        pure_prng(prng_type = 'xoshiro256++') -> Set the pseudo-random number generator algorithm used.
        
        Note
        ----
        The generated instance is thread-safe.
        
        Only the pseudo-random number generation algorithm with period of 2^n or 2^n-1 is adapted.
        
        The pseudo-random number generation algorithm implemented here must be the full-period length output.
    '''
    
    VERSION: Final = '0.8.3'
    
    prng_algorithms_list = ['xoshiro256++']
    
    prng_period_dict = {'xoshiro256++': 2 ** 256 - 1}
    seed_length_dict = {'xoshiro256++': 256}
    
    def __init__(self, seed: Optional[int] = None, prng_type: str = 'xoshiro256++') -> None:
        '''
            Set up an instance of a pseudorandom number generator.
            
            Parameters
            ----------
            seed : integer, or None (default)
                Sets the seed for the current instance.
            
            prng_type: str, or 'xoshiro256++' (default)
                Set the pseudo-random number algorithm used for the current instance.
        '''
        self.prng_algorithms_dict = {'xoshiro256++': self.__xoshiro256plusplus}
        
        assert isinstance(seed, (int, type(None))), 'Error: The value of the seed is non-integer.'
        if isinstance(seed, int): assert seed >= 0, "Error: seed can't be negative."
        assert prng_type in self.prng_algorithms_dict.keys(), 'Error: The string for prng_type is not in the specified list.'
        
        self.seed = self.__seed_initialization(seed, prng_type)
        self.prng_type = prng_type
        self.prev_new_period: Optional[int] = None
    
    def __seed_initialization(self, seed: Union[int, None], prng_type: str) -> int:  #The original seed is hash obfuscated for pseudo-random generation.
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
    def __xoshiro256plusplus(self) -> int:  #Xoshiro256PlusPlus method implements full-period length output, [1, 2^ 256-1]
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
            The value type and value range are determined by the random algorithm, which is specified by parameter `prng_type` at instance initialization.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.source_random_number()
            73260932800743358445652462028207907455677987852735468159219395093090100006110
        '''
        return self.prng_algorithms_dict[self.prng_type]()
    
    def rand_float(self, new_period: Optional[int] = None) -> gmpy2.mpfr:
        '''
            Parameters
            ----------
            new_period : integer
                Set the period of the pseudo-random sequence.
            
            Returns a random float in [0, 1), with 0 included and 1 excluded.
            The return type is `gmpy2.mpfr`.
            The floating-point length is the output length of the specified algorithm plus one.
            
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.rand_float()
            mpfr('0.6326937641706669741872583730940429737405414921354622618051716414693676562568086',257)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.rand_float(period)
            mpfr('0.02795744845257346733436109648463446736744766610965612207643215290679786849298934',256)
        '''
        
        if new_period is None:
            prng_period = self.__class__.prng_period_dict[self.prng_type]
            prng_algorithm_lower = prng_period & 1
        else:
            prng_period = new_period
            prng_algorithm_lower = 0
        prng_period_bit_size = prng_period.bit_length() - 1 + prng_algorithm_lower
        with gmpy2.local_context(gmpy2.context(), precision = prng_period_bit_size + 1):
            if new_period is None:
                random_number = gmpy2.mpfr(self.source_random_number() - prng_algorithm_lower)
            else:
                random_number = gmpy2.mpfr(self.rand_with_period(prng_period))
            return random_number / gmpy2.mpfr(prng_period)
    
    def rand_int(self, b: int, a: int = 0, new_period: Optional[int] = None) -> gmpy2.mpz:
        '''
            Parameters
            ----------
            b : integer
                Upper bound on the range including 'b'.
                
            a : integer, or 0 (default)
                Lower bound on the range including 'a'.
            
            new_period : integer
                Set the period of the pseudo-random sequence.
            
            Returns an integer pseudo-random number in the range [a, b].
            
            Note
            ----
            The scale from a to b cannot exceed the period of the pseudo-random number generator.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.rand_int(100, 1)
            mpz(94)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.rand_int(100, 1, period)
            mpz(38)
        '''
        assert isinstance(b, int), 'Error: The value of b is non-integer.'
        assert isinstance(a, int), 'Error: The value of a is non-integer.'
        
        difference_value = b - a
        if new_period is None:
            prng_period = self.__class__.prng_period_dict[self.prng_type]
            prng_algorithm_lower = prng_period & 1
        else:
            prng_period = new_period
            prng_algorithm_lower = 0
        assert difference_value < prng_period, 'Error: The a to b scale extends beyond the period of the pseudo-random number generator.'
        
        scale = difference_value + prng_algorithm_lower
        bit_mask = gmpy2.bit_mask(scale.bit_length())
        if new_period is None:
            random_number_masked = self.source_random_number() & bit_mask
            while not (prng_algorithm_lower <= random_number_masked <= scale):
                random_number_masked = self.source_random_number() & bit_mask
        else:
            random_number_masked = self.rand_with_period(prng_period) & bit_mask
            while not (random_number_masked <= scale):
                random_number_masked = self.rand_with_period(prng_period) & bit_mask
        return a + (random_number_masked - prng_algorithm_lower)
    
    def get_randint_set(self, b: int, a: int, k: int) -> Set[gmpy2.mpz]:
        '''
            Parameters
            ----------
            b : integer
                Upper bound on the range including 'b'.
                
            a : integer
                Lower bound on the range including 'a'.
                
            k : integer
                The number of set elements to generate.
            
            Returns a set of pseudo-random Numbers with k elements in the range [a, b].
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.get_randint_set(100, 1, 6)
            {mpz(98), mpz(68), mpz(46), mpz(24), mpz(27), mpz(94)}
        '''
        assert isinstance(a, int), 'Error: The value of a is non-integer.'
        assert isinstance(b, int), 'Error: The value of b is non-integer.'
        assert isinstance(k, int), 'Error: The value of k is non-integer.'
        assert a <= b, 'Error: a cannot be greater than b.'
        assert k >= 0, "Error: k can't be negative."
        assert k <= (b - a), "Error: k can't be greater than b minus a."
        
        randint_set: Set[gmpy2.mpz] = set()
        while len(randint_set) < k:
            randint_set.add(self.rand_int(b, a))
        return randint_set
    
    def rand_with_period(self, new_period: int) -> int:
        '''
            Generates an integer pseudo-random number with a specified period.
            
            Parameters
            ----------
            new_period : integer
                Set the period of the pseudo-random sequence.
            
            Returns a pseudo-random integer for a new period.
            
            Note
            ----
            Set new_period to be no less than 2.
            
            The value of (new period / original period) is the representation of generating efficiency.
When the difference between the new period and the original period is too large, it may takes a **long** time to generate a pseudo-random number!
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.rand_with_period(period)
            40688839126177430252467309162469901643963863918059158449302074429100738061310
        '''
        from bisect import bisect_left
        
        assert isinstance(new_period, int), 'Error: The value of new_period is non-integer.'
        prng_period = self.__class__.prng_period_dict[self.prng_type]
        assert new_period >= 2, 'Error: new_period cannot be less than 2.'
        assert new_period <= prng_period, 'Error: Suppose the new period number cannot be greater than the original period number of the pseudorandom number generator.'
        
        prng_algorithm_lower = prng_period & 1
        number_to_subtract = prng_period - new_period
        
        if new_period != self.prev_new_period:
            self.set_of_numbers_to_exclude = self.get_randint_set(prng_period, prng_algorithm_lower, number_to_subtract)
            self.ordered_list_of_excluded_numbers = sorted(self.set_of_numbers_to_exclude)
            self.prev_new_period = new_period
        
        random_number = self.source_random_number()
        assert isinstance(random_number, int), 'Error: The chosen pseudo-random number algorithm is non-integer.'
        while True:
            if random_number not in self.set_of_numbers_to_exclude:
                break
            random_number = self.source_random_number()
        
        random_number -= bisect_left(self.ordered_list_of_excluded_numbers, random_number)
        return random_number - prng_algorithm_lower
