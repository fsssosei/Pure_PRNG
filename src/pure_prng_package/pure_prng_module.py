'''
pure_prng - This package is used to generate professional pseudo-random Numbers.
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

from typing import Optional, Union, Set
from collections import namedtuple
import gmpy2
from pure_nrng_package import *

__all__ = ['pure_prng']

class pure_prng:
    '''
        Generate multi-precision pseudo-random Numbers.
        There are "methods" that specify the period of a pseudo-random sequence.
        
        pure_prng() -> The system random number as the seed of the pseudo-random number generator.
        
        pure_prng(seed) -> A pseudo-random number generator that specifies seeds.
        
        pure_prng(prng_type = {default_prng_type}) -> Set the pseudo-random number generator algorithm used.
        
        Note
        ----
        The generated instance is thread-safe.
        
        Only the pseudo-random number generation algorithm with period of 2^n or 2^n-1 is adapted.
    '''
    
    version = '0.9.4'
    
    hash_algorithm_argument = namedtuple('hash_algorithm_argument', ('period', 'seed_size', 'output_size'))
    hash_algorithms_dict = {'xoshiro256++': hash_algorithm_argument(period = 2 ** 256 - 1, seed_size = 256, output_size = 256)}
    
    prng_type_list = list(hash_algorithms_dict.keys())
    default_prng_type = 'xoshiro256++'
    __doc__ = __doc__.replace('{default_prng_type}', default_prng_type)
    
    def __init__(self, seed: Optional[int] = None, prng_type: str = default_prng_type) -> None:
        '''
            Create an instance of a pseudo-random number generator.  创建一个伪随机数生成器的实例。
            
            Parameters
            ----------
            seed: int, default None
                Sets the seed for the current instance.
            
            prng_type: str, default {default_prng_type}
                Set the pseudo-random number algorithm used for the current instance.
                Available algorithms: {prng_type_list}
        '''
        hash_callable_tuples = namedtuple('hash_callable_tuples', ('seed_init_callable', 'hash_callable'))
        self.hash_callable_dict = {'xoshiro256++': hash_callable_tuples(seed_init_callable = self.__seed_initialize_xoshiro256plusplus, hash_callable = self.__xoshiro256plusplus)}
        
        assert isinstance(seed, (int, type(None))), f'seed must be an int or None, got type {type(seed).__name__}'
        if isinstance(seed, int) and (seed < 0): raise ValueError('seed must be >= 0')
        if prng_type not in self.hash_callable_dict.keys(): raise ValueError('The string for prng_type is not in the list of implemented algorithms.')
        
        self.prng_type = prng_type
        self.__seed_initialization(seed, prng_type)
        self.prev_new_period: Optional[int] = None
    __init__.__doc__ = __init__.__doc__.replace('{default_prng_type}', default_prng_type)
    __init__.__doc__ = __init__.__doc__.replace('{prng_type_list}', ', '.join([item for item in prng_type_list]))
    
    
    def __seed_initialization(self, seed: Union[int, None], prng_type: str) -> None:  #The original seed is hash obfuscated for pseudo-random generation.
        seed_size = self.__class__.hash_algorithms_dict[prng_type].seed_size
        
        nrng_instance = pure_nrng()
        if seed is None:
            seed = nrng_instance.true_rand_bits(seed_size)  #Read unreproducible seeds provided by the operating system.
        else:
            seed = rng_util.randomness_extractor(seed, seed_size)
        
        self.hash_callable_dict[prng_type].seed_init_callable(locals(), seed, seed_size)  #The specific initialization seed method is called according to prng_type.
    
    
    def __seed_initialize_xoshiro256plusplus(self, seed_init_locals: dict, seed: int, seed_size: int) -> None:  #Generate the self variable used by the Xoshiro256PlusPlus algorithm.
        from math import ceil
        from hashlib import blake2b
        from struct import unpack
        
        seed_byte_length = ceil(seed_size / 8)  #Converts bit length to byte length.
        blake2b_digest_size = seed_byte_length
        
        full_zero_bytes = (0).to_bytes(blake2b_digest_size, byteorder = 'little')
        while True:  #The Xoshiro256PlusPlus algorithm requires that the input seed value is not zero.
            hash_seed_bytes = blake2b(seed.to_bytes(seed_byte_length, byteorder = 'little'), digest_size = blake2b_digest_size).digest()
            if hash_seed_bytes == full_zero_bytes:  #Avoid hash results that are zero.
                seed += 1  #Changing the seed value to produce a new hash result.
                seed = rng_util.bit_length_mask(seed, seed_size)
            else:
                break
        self.s_array_of_xoshiro256plusplus = list(unpack('<QQQQ', hash_seed_bytes))  #Xoshiro256PlusPlus to use the 256 bit uint64 seed list.
    
    
    def __xoshiro256plusplus(self) -> int:  #Xoshiro256PlusPlus method implements full-period length output, [1, 2^ 256-1]
        #The external variable used is "self.s_array_of_xoshiro256plusplus".
        mask64 = 18446744073709551615  #(2 ** 64) - 1
        
        def rotl64(x: int, k: int) -> int:
            return ((x << k) & mask64) | (x >> (64 - k))
        
        result = 0
        for i in range(4):
            result |= (rotl64(self.s_array_of_xoshiro256plusplus[0] + self.s_array_of_xoshiro256plusplus[3], 23) + self.s_array_of_xoshiro256plusplus[0]) << (64 * i)
            t = (self.s_array_of_xoshiro256plusplus[1] << 17) & mask64
            self.s_array_of_xoshiro256plusplus[2] ^= self.s_array_of_xoshiro256plusplus[0]
            self.s_array_of_xoshiro256plusplus[3] ^= self.s_array_of_xoshiro256plusplus[1]
            self.s_array_of_xoshiro256plusplus[1] ^= self.s_array_of_xoshiro256plusplus[2]
            self.s_array_of_xoshiro256plusplus[0] ^= self.s_array_of_xoshiro256plusplus[3]
            self.s_array_of_xoshiro256plusplus[2] ^= t
            self.s_array_of_xoshiro256plusplus[3] = rotl64(self.s_array_of_xoshiro256plusplus[3], 45)
        return result
    
    
    def source_random_number(self) -> Union[int, float]:
        '''
            The source random number directly derived from the random generator algorithm.
            The result can be processed into other data types in other methods.
            
            Returns
            -------
            source_random_number: int, or float
                Returns a pseudo-random number.
            
            Note
            ----
            The value type and value range are determined by the random algorithm, which is specified by parameter 'prng_type' at instance initialization.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.source_random_number()
            63704397730169193686456860639078459647664747236380824242857347684562650854070
        '''
        return self.hash_callable_dict[self.prng_type].hash_callable()
    
    
    def rand_float(self, new_period: Optional[int] = None) -> gmpy2.mpfr:
        '''
            Generate a pseudo-random real number (you can set the pseudo-random number generator algorithm period).  生成一个伪随机实数（可设定伪随机数生成器算法周期）。
            
            Parameters
            ----------
            new_period: int, default None
                Set the period of the pseudo-random sequence.
                The default is not to change the eigenperiod of the selected pseudo-random number generator algorithm. 缺省就是不改变所选伪随机数生成算法的本征周期。
            
            Returns
            -------
            rand_float: gmpy2.mpfr
                Returns a pseudo-random real number in [0, 1), with 0 included and 1 excluded.
                The floating-point length is the output length of the specified algorithm plus one.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.rand_float()
            mpfr('0.5501619164985171033237722626311247973459654767593884512471329918122925574973017',257)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.rand_float(period)
            mpfr('0.6665079772632617788674079157248027245703466196226109430388828957294865649611888',256)
        '''
        assert isinstance(new_period, (int, type(None))), f'new_period must be an int or None, got type {type(new_period).__name__}'
        
        if new_period is None:
            prng_period = self.__class__.hash_algorithms_dict[self.prng_type].period
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
            Generates a pseudo-random integer within a specified interval (can set the pseudo-random number generator algorithm period).  生成一个指定区间内的伪随机整数（可设定伪随机数生成器算法周期）。
            
            Parameters
            ----------
            b: int
                Upper bound on the range including 'b'.
            
            a: int, default 0
                Lower bound on the range including 'a'.
            
            new_period: int, default None
                Set the period of the pseudo-random sequence.
            
            Returns
            -------
            rand_int: gmpy2.mpz
                Returns an integer pseudo-random number in the range [a, b].
            
            Note
            ----
            The scale from a to b cannot exceed the period of the pseudo-random number generator.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.rand_int(100, 1)
            mpz(54)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.rand_int(100, 1, period)
            mpz(61)
        '''
        assert isinstance(b, int), f'b must be an int, got type {type(b).__name__}'
        assert isinstance(a, int), f'a must be an int, got type {type(a).__name__}'
        assert isinstance(new_period, (int, type(None))), f'new_period must be an int or None, got type {type(new_period).__name__}'
        
        difference_value = b - a
        if new_period is None:
            prng_period = self.__class__.hash_algorithms_dict[self.prng_type].period
            prng_algorithm_lower = prng_period & 1
        else:
            prng_period = new_period
            prng_algorithm_lower = 0
        if difference_value >= prng_period: raise ValueError('The a to b scale extends beyond the period of the pseudo-random number generator.')
        
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
            Generates a set of pseudo-random integers in a specified interval. 生成一个指定区间内的伪随机整数的集合。
            
            Parameters
            ----------
            b: int
                Upper bound on the range including 'b'.
            
            a: int
                Lower bound on the range including 'a'.
            
            k: int
                The number of set elements to generate.
            
            Returns
            -------
            get_randint_set: Set[gmpy2.mpz]
                Returns a set of pseudo-random Numbers with k elements in the range [a, b].
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> prng_instance.get_randint_set(100, 1, 6)
            {mpz(5), mpz(9), mpz(18), mpz(54), mpz(57), mpz(93)}
        '''
        assert isinstance(b, int), f'b must be an int, got type {type(b).__name__}'
        assert isinstance(a, int), f'a must be an int, got type {type(a).__name__}'
        assert isinstance(k, int), f'k must be an int, got type {type(k).__name__}'
        if a > b: raise ValueError('a must be <= b')
        if k < 0: raise ValueError('k must be >= 0')
        if k > (b - a): raise ValueError("k can't be greater than b minus a")
        
        randint_set: Set[gmpy2.mpz] = set()
        while len(randint_set) < k:
            randint_set.add(self.rand_int(b, a))
        return randint_set
    
    
    def rand_with_period(self, new_period: int) -> int:
        '''
            Generates an integer pseudo-random number with a specified period.
            
            Parameters
            ----------
            new_period: int
                Set the period of the pseudo-random sequence.
            
            Returns
            -------
            rand_with_period: int
                Returns a pseudo-random integer for a new period.
            
            Note
            ----
            Set new_period to be no less than 2.
            
            Return value is the range of [0, new_period), with zero included and new_period excluded.
            
            The value of (new period / original period) is the representation of generating efficiency.
            When the difference between the new period and the original period is too large, it may takes a long time to generate a pseudo-random number!
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> prng_instance.rand_with_period(period)
            103085265137502064472166298218885841110988755115459404830932952476483720814169
        '''
        assert isinstance(new_period, int), f'new_period must be an int, got type {type(new_period).__name__}'
        prng_period = self.__class__.hash_algorithms_dict[self.prng_type].period
        if new_period < 2: raise ValueError('new_period must be >= 2')
        if new_period > prng_period: raise ValueError('Suppose the new period number cannot be greater than the original period number of the pseudorandom number generator.')
        
        from bisect import bisect_left
        
        prng_algorithm_lower = prng_period & 1
        number_to_subtract = prng_period - new_period
        
        if new_period != self.prev_new_period:
            self.set_of_numbers_to_exclude = self.get_randint_set(prng_period, prng_algorithm_lower, number_to_subtract)
            self.ordered_list_of_excluded_numbers = sorted(self.set_of_numbers_to_exclude)
            self.prev_new_period = new_period
        
        random_number = self.source_random_number()
        assert isinstance(random_number, int), 'The chosen pseudo-random number algorithm is non-integer.'
        while True:
            if random_number not in self.set_of_numbers_to_exclude:
                break
            random_number = self.source_random_number()
        
        random_number -= bisect_left(self.ordered_list_of_excluded_numbers, random_number)
        return random_number - prng_algorithm_lower
