'''
pure_prng - This package is used to generate professional pseudo-random Numbers.
Copyright (C) 2020-2021  sosei
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

from typing import TypeVar, Optional, Union, Callable, Set, Iterator
from hashlib import shake_256
from bisect import bisect_left
from gmpy2 import mpfr, mpz, local_context as gmpy2_local_context, context as gmpy2_get_context, bit_mask as gmpy2_bit_mask, c_div as gmpy2_c_div, t_div as gmpy2_t_div, t_divmod as gmpy2_t_divmod
from randomgen import PCG64, LCG128Mix, EFIIX64, Philox as PhiloxCounter, ThreeFry as ThreeFryCounter, AESCounter, ChaCha as ChaChaCounter, SPECK128 as SPECKCounter
from prng_algorithms_package import *
from pure_nrng_package import *

__all__ = ['pure_prng']

Integer = TypeVar('Integer', int, mpz)

Raw_binary_number = mpz
Randomized_consecutive_numbers = mpz

class pure_prng:
    '''
        Generate multi-precision pseudo-random Numbers.
        There are "methods" that specify the period of a pseudo-random sequence.
        Only the pseudo-random number algorithm with good statistical properties is implemented.
        
        Note
        ----
        The generated instance is thread-safe.
        
        It must be a pseudo-random number generation algorithm with hash block values in [0, 2^n-1], and k-dimensional uniform distribution.  必须是hash块值域在[0, 2^n-1]，和k维均匀分布的伪随机数生成算法。
        The period of the available algorithm must be an integer multiple of 2^hash_size.  可用算法的周期必须是2^hash_size的整数倍。
        
        List of PRNG algorithms                  Period
        -----------------------                  ------
        Quadratic Congruential Generator(QCG)    2^256
        Cubic Congruential Generator(CCG)        2^256
        Inversive Congruential Generator(ICG)    102*2^256
        PCG64_XSL_RR                             2^128
        PCG64_DXSM                               2^128
        LCG64_32_ext                             2^128
        LCG128Mix_XSL_RR                         2^128
        LCG128Mix_DXSM                           2^128
        LCG128Mix_MURMUR3                        2^128
        PhiloxCounter                            4*2^(4*64)
        ThreeFryCounter                          4*2^(4*64)
        AESCounter                               2^128
        ChaChaCounter                            2^128
        SPECKCounter                             2^129
        XSM64                                    2^128
        EFIIX64                                  2^64
        SplitMix64                               2^64
        Ran64                                    2^64
    '''
    
    version = '2.9.0'
    
    prng_algorithms_dict = {'QCG': {'hash_period': 1 << 256, 'variable_period': True, 'additional_hash': True, 'seed_range': 1 << 256, 'hash_size': 256},
                            'CCG': {'hash_period': 1 << 256, 'variable_period': True, 'additional_hash': True, 'seed_range': 1 << 256, 'hash_size': 256},
                            'ICG': {'hash_period': (1 << 256) * 102, 'variable_period': False, 'additional_hash': False, 'seed_range': 2 ** 256 * 102 + 1, 'hash_size': 256},
                            'PCG64_XSL_RR': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'PCG64_DXSM': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'LCG64_32_ext': {'hash_period': 1 << 128, 'variable_period': True, 'additional_hash': False, 'seed_range': 1 << 64, 'hash_size': 32},
                            'LCG128Mix_XSL_RR': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'LCG128Mix_DXSM': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'LCG128Mix_MURMUR3': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'PhiloxCounter': {'hash_period': 4 * 2 ** (4 * 64), 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'ThreeFryCounter': {'hash_period': 4 * 2 ** (4 * 64), 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'AESCounter': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 128, 'hash_size': 64},
                            'ChaChaCounter': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 256, 'hash_size': 64},
                            'SPECKCounter': {'hash_period': 1 << 129, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 256, 'hash_size': 64},
                            'XSM64': {'hash_period': 1 << 128, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 64, 'hash_size': 64},
                            'EFIIX64': {'hash_period': 1 << 64, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 64, 'hash_size': 64},
                            'SplitMix64': {'hash_period': 1 << 64, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 64, 'hash_size': 64},
                            'Ran64': {'hash_period': 1 << 64, 'variable_period': False, 'additional_hash': False, 'seed_range': 1 << 64, 'hash_size': 64}}
    
    for _, algorithm_characteristics_parameter in prng_algorithms_dict.items():
        hash_period = algorithm_characteristics_parameter['hash_period']
        if hash_period == float('+inf'):
            algorithm_characteristics_parameter['prng_period'] = float('+inf')
            algorithm_characteristics_parameter['output_width'] = 1
            algorithm_characteristics_parameter['output_size'] = hash_size
        else:
            period_bit_length = mpz(hash_period - 1).num_digits(2)
            hash_size = algorithm_characteristics_parameter['hash_size']
            output_width = gmpy2_c_div(period_bit_length, hash_size)
            algorithm_characteristics_parameter['prng_period'] = gmpy2_t_div(hash_period, output_width)
            algorithm_characteristics_parameter['output_width'] = output_width
            algorithm_characteristics_parameter['output_size'] = hash_size * output_width
    
    prng_type_list = list(prng_algorithms_dict.keys())
    default_prng_type = 'QCG'
    #__doc__ = __doc__.replace('', '')
    
    def __init__(self, seed: Optional[Integer] = None, prng_type: str = default_prng_type, new_prng_period: Optional[Integer] = None, additional_hash: Union[bool, Callable[[int, int], int], None] = None) -> None:
        '''
            Create an instance of a pseudo-random number generator.  创建一个伪随机数生成器的实例。
            
            Parameters
            ----------
            seed: Integer, default None
                Sets the seed for the current instance.
                By default, the random number of the system is used as the seed for the pseudorandom number generator.
            
            prng_type: str, default {default_prng_type}
                Set the pseudo-random number algorithm used for the current instance.
                Available algorithms: {prng_type_list}
            
            new_prng_period: Integer, default None
                Set the period for a variable period hash. ('variable_period' is true pseudo-random number generator algorithm)
                The period that is actually set will be >= the input new_prng_period.
            
            additional_hash: bool, or Callable[[int, int], int]], default None
                The default is to be confused by the built-in Settings of each algorithm.  缺省是按每种算法内置设定决定混淆与否。
                Set to Boolean to force the current algorithm to enable built-in security hashing to further confuse pseudo-random Numbers.  设为布尔量来强制决定当前算法是否启用内置安全散列对伪随机数做进一步混淆。
                Or introduce an external hash function to accomplish this.  或引入外部散列函数完成此功能。
        '''
        assert isinstance(seed, (int, type(mpz(0)), type(None))), f'seed must be an Integer or None, got type {type(seed).__name__}'
        assert isinstance(prng_type, str), f'prng_type must be an str, got type {type(prng_type).__name__}'
        assert isinstance(new_prng_period, (int, type(mpz(0)), type(None))), f'new_prng_period must be an Integer or None, got type {type(new_prng_period).__name__}'
        assert isinstance(additional_hash, (bool, Callable, type(None))), f'additional_hash must be an bool or Callable or None, got type {type(additional_hash).__name__}'
        
        all_hash_callable_dict = {'internal_generator': {'seed_init_callable': self.__seed_initialize_internal_generator, 'set_hash_period_callable': self.__set_hash_period_of_general, 'hash_callable': self.__internal_generator},
                                  'QCG': {'seed_init_callable': self.__seed_initialize_quadratic_congruential_generator, 'set_hash_period_callable': self.__set_hash_period_of_general, 'hash_callable': self.__quadratic_congruential_generator},
                                  'CCG': {'seed_init_callable': self.__seed_initialize_cubic_congruential_generator, 'set_hash_period_callable': self.__set_hash_period_of_general, 'hash_callable': self.__cubic_congruential_generator},
                                  'ICG': {'seed_init_callable': self.__seed_initialize_icg, 'hash_callable': self.__icg},
                                  'PCG64_XSL_RR': {'seed_init_callable': self.__seed_initialize_pcg64_xsl_rr, 'hash_callable': self.__pcg64_xsl_rr},
                                  'PCG64_DXSM': {'seed_init_callable': self.__seed_initialize_pcg64_dxsm, 'hash_callable': self.__pcg64_dxsm},
                                  'LCG64_32_ext': {'seed_init_callable': self.__seed_initialize_lcg64_32_ext, 'set_hash_period_callable': self.__set_hash_period_of_lcg64_32_ext, 'hash_callable': self.__lcg64_32_ext},
                                  'LCG128Mix_XSL_RR': {'seed_init_callable': self.__seed_initialize_lcg128mix_xsl_rr, 'hash_callable': self.__lcg128mix_xsl_rr},
                                  'LCG128Mix_DXSM': {'seed_init_callable': self.__seed_initialize_lcg128mix_dxsm, 'hash_callable': self.__lcg128mix_dxsm},
                                  'LCG128Mix_MURMUR3': {'seed_init_callable': self.__seed_initialize_lcg128mix_murmur3, 'hash_callable': self.__lcg128mix_murmur3},
                                  'PhiloxCounter': {'seed_init_callable': self.__seed_initialize_philox_counter, 'hash_callable': self.__philox_counter},
                                  'ThreeFryCounter': {'seed_init_callable': self.__seed_initialize_threefry_counter, 'hash_callable': self.__threefry_counter},
                                  'AESCounter': {'seed_init_callable': self.__seed_initialize_aes_counter, 'hash_callable': self.__aes_counter},
                                  'ChaChaCounter': {'seed_init_callable': self.__seed_initialize_chacha_counter, 'hash_callable': self.__chacha_counter},
                                  'SPECKCounter': {'seed_init_callable': self.__seed_initialize_speck_counter, 'hash_callable': self.__speck_counter},
                                  'XSM64': {'seed_init_callable': self.__seed_initialize_xsm64, 'hash_callable': self.__xsm64},
                                  'EFIIX64': {'seed_init_callable': self.__seed_initialize_efiix64, 'hash_callable': self.__efiix64},
                                  'SplitMix64': {'seed_init_callable': self.__seed_initialize_splitmix64, 'hash_callable': self.__splitmix64},
                                  'Ran64': {'seed_init_callable': self.__seed_initialize_ran64, 'hash_callable': self.__ran64}}
        
        if (seed is not None) and (seed < 0): raise ValueError('seed must be >= 0')
        prng_algorithms_dict = self.__class__.prng_algorithms_dict
        if prng_type not in prng_algorithms_dict.keys(): raise ValueError('The string for prng_type is not in the list of implemented algorithms.')
        
        algorithm_characteristics_parameter = prng_algorithms_dict[prng_type]
        if algorithm_characteristics_parameter['variable_period']:
            if new_prng_period is not None:
                if new_prng_period < 1: raise ValueError('new_prng_period must be >= 1')
                if new_prng_period & gmpy2_bit_mask(mpz(new_prng_period).num_digits(2) - 1) != 0: raise ValueError('new_prng_period must be a power of 2')
                self.__set_hash_period(algorithm_characteristics_parameter, new_prng_period, all_hash_callable_dict[prng_type]['set_hash_period_callable'])
        else:
            if new_prng_period is not None: raise TypeError(f'The {prng_type} algorithm cannot modify the period.')
        
        if algorithm_characteristics_parameter['hash_period'] != float('+inf'):
            internal_algorithm_characteristics_parameter = dict()
            internal_output_size = algorithm_characteristics_parameter['output_size']
            internal_algorithm_characteristics_parameter['hash_period'] = 1 << internal_output_size
            internal_algorithm_characteristics_parameter['variable_period'] = True
            internal_algorithm_characteristics_parameter['additional_hash'] = False
            internal_algorithm_characteristics_parameter['seed_range'] = 1 << internal_output_size
            internal_algorithm_characteristics_parameter['hash_size'] = internal_output_size
            internal_algorithm_characteristics_parameter['prng_period'] = 1 << internal_output_size
            internal_algorithm_characteristics_parameter['output_width'] = 1
            internal_algorithm_characteristics_parameter['output_size'] = internal_output_size
            prng_algorithms_dict['internal_generator'] = internal_algorithm_characteristics_parameter
        
        if additional_hash is None:
            if algorithm_characteristics_parameter['additional_hash']:
                all_hash_callable_dict[prng_type]['additional_hash_callable'] = rng_util.randomness_extractor
            else:
                all_hash_callable_dict[prng_type]['additional_hash_callable'] = None
        elif additional_hash in (True, False):
            if additional_hash:
                all_hash_callable_dict[prng_type]['additional_hash_callable'] = rng_util.randomness_extractor
            else:
                all_hash_callable_dict[prng_type]['additional_hash_callable'] = None
        else:
            all_hash_callable_dict[prng_type]['additional_hash_callable'] = additional_hash
        
        hash_callable_dict = all_hash_callable_dict[prng_type]
        self.__seed_initialization(seed, algorithm_characteristics_parameter, hash_callable_dict['seed_init_callable'])
        hash_callable_dict['hash_callable'] = hash_callable_dict['hash_callable'](algorithm_characteristics_parameter)  #Changed the all_hash_callable_dict
        
        all_hash_callable_dict['internal_generator']['additional_hash_callable'] = None
        hash_callable_dict = all_hash_callable_dict['internal_generator']
        self.__seed_initialization(seed, internal_algorithm_characteristics_parameter, hash_callable_dict['seed_init_callable'])
        hash_callable_dict['hash_callable'] = hash_callable_dict['hash_callable'](internal_algorithm_characteristics_parameter)  #Changed the all_hash_callable_dict
        
        self.prng_type = prng_type
        self.all_hash_callable_dict = all_hash_callable_dict
    __init__.__doc__ = __init__.__doc__.replace('{default_prng_type}', default_prng_type)
    __init__.__doc__ = __init__.__doc__.replace('{prng_type_list}', ', '.join([item for item in prng_type_list]))
    
    
    def __set_hash_period(self, algorithm_characteristics_parameter: dict, new_prng_period: Integer, set_hash_period_callable: Callable[[dict, Integer], None]) -> None:
        set_hash_period_callable(algorithm_characteristics_parameter, new_prng_period)
    
    
    def __set_hash_period_of_general(self, algorithm_characteristics_parameter: dict, new_prng_period: Integer) -> None:
        new_prng_period_bit_length = mpz(new_prng_period - 1).num_digits(2)
        algorithm_characteristics_parameter['hash_period'] = new_prng_period
        algorithm_characteristics_parameter['seed_range'] = 1 << new_prng_period_bit_length
        algorithm_characteristics_parameter['hash_size'] = new_prng_period_bit_length
        algorithm_characteristics_parameter['prng_period'] = new_prng_period
        algorithm_characteristics_parameter['output_width'] = 1
        algorithm_characteristics_parameter['output_size'] = new_prng_period_bit_length
    
    
    def __set_hash_period_of_lcg64_32_ext(self, algorithm_characteristics_parameter: dict, new_prng_period: Integer) -> None:
        new_prng_period_bit_length = mpz(new_prng_period - 1).num_digits(2)
        
        q, r = divmod(new_prng_period_bit_length, 32)
        output_width = q + (1 if r != 0 else 0)
        new_hash_period_bit_length =  new_prng_period_bit_length * output_width
        
        q, r = divmod(new_hash_period_bit_length, 32)
        n = mpz(q - 2).num_digits(2) - (1 if r == 0 else 0)
        set_hash_period_bit_length = 32 * (2 ** n + 2)
        
        hash_period = 2 ** set_hash_period_bit_length
        hash_size = algorithm_characteristics_parameter['hash_size']
        
        algorithm_characteristics_parameter['hash_period'] = hash_period
        algorithm_characteristics_parameter['prng_period'] = gmpy2_t_div(hash_period, output_width)
        algorithm_characteristics_parameter['output_width'] = output_width
        algorithm_characteristics_parameter['output_size'] = hash_size * output_width
    
    
    def __seed_initialization(self, seed: Union[Integer, None], algorithm_characteristics_parameter: dict, hash_callable: Callable[[dict, Integer, dict], None]) -> None:  #The original seed is hash obfuscated for pseudo-random generation.
        seed_range = algorithm_characteristics_parameter['seed_range']
        
        if seed is None:
            nrng_instance = pure_nrng()
            nrng_instance_true_rand_bits = nrng_instance.true_rand_bits(mpz(seed_range - 1).num_digits(2))
            seed = next(nrng_instance_true_rand_bits)  #Read unreproducible seeds provided by the operating system.
        else:
            seed = seed % seed_range
        seed = int(seed)
        hash_callable(locals(), seed, algorithm_characteristics_parameter)  #The specific initialization seed method is called according to prng_type.
    
    
    def __seed_initialize_internal_generator(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #Generate the state variables used by the internal algorithm.
        self.state_of_ig = seed
    
    
    def __internal_generator(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.state_of_ig".
        x = self.state_of_ig
        
        m = algorithm_characteristics_parameter['hash_size']
        while True:
            x = rng_util.bit_length_mask(((x**3)<<2) + ((x**2)<<1) + ((x<<2)-x) + 1, m)
            yield x
    
    
    def __seed_initialize_quadratic_congruential_generator(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #Generate the state variables used by the QCG algorithm.
        self.state_of_qcg = seed
    
    
    def __quadratic_congruential_generator(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.state_of_qcg".
        x = self.state_of_qcg
        
        m = algorithm_characteristics_parameter['hash_size']
        for _ in range(algorithm_characteristics_parameter['hash_period']):
            x = rng_util.bit_length_mask(((x**2)<<1) + ((x<<2)-x) + 1, m)
            yield x
        raise StopIteration('The number of times it is generated exceeds the number of hash period.')  #生成次数超出散列周期数。
    
    
    def __seed_initialize_cubic_congruential_generator(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #Generate the state variables used by the CCG algorithm.
        self.state_of_ccg = seed
    
    
    def __cubic_congruential_generator(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.state_of_ccg".
        x = self.state_of_ccg
        
        m = algorithm_characteristics_parameter['hash_size']
        for _ in range(algorithm_characteristics_parameter['hash_period']):
            x = rng_util.bit_length_mask(((x**3)<<2) + ((x**2)<<1) + ((x<<2)-x) + 1, m)
            yield x
        raise StopIteration('The number of times it is generated exceeds the number of hash period.')  #生成次数超出散列周期数。
    
    
    def __seed_initialize_icg(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The ICG method is initialized with seeds.
        self.icg_instance = ICG(seed)
    
    
    def __icg(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.icg_instance".
        while True:
            yield self.icg_instance.random_raw()
    
    
    def __seed_initialize_pcg64_xsl_rr(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The PCG64_XSL_RR method is initialized with seeds.
        self.pcg64_xsl_rr_instance = PCG64(seed, inc = None, variant = 'xsl-rr', mode = "sequence")
    
    
    def __pcg64_xsl_rr(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.pcg64_xsl_rr_instance".
        while True:
            yield self.pcg64_xsl_rr_instance.random_raw()
    
    
    def __seed_initialize_pcg64_dxsm(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The PCG64_DXSM method is initialized with seeds.
        self.pcg64_dxsm_instance = PCG64(seed, inc = None, variant = 'dxsm', mode = "sequence")
    
    
    def __pcg64_dxsm(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.pcg64_dxsm_instance".
        while True:
            yield self.pcg64_dxsm_instance.random_raw()
    
    
    def __seed_initialize_lcg64_32_ext(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The LCG64_32_ext method is initialized with seeds.
        period_bit_length = mpz(algorithm_characteristics_parameter['hash_period'] - 1).num_digits(2)
        q = period_bit_length // 32
        n = mpz(q - 2).num_digits(2) - 1
        self.lcg64_32_ext_instance = LCG64_32_ext(seed, n)
    
    
    def __lcg64_32_ext(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.lcg64_32_ext_instance".
        while True:
            yield self.lcg64_32_ext_instance.random_raw()
    
    
    def __seed_initialize_lcg128mix_xsl_rr(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The LCG128Mix_XSL_RR method is initialized with seeds.
        self.lcg128mix_xsl_rr_instance = LCG128Mix(seed, output = 'xsl-rr')
    
    
    def __lcg128mix_xsl_rr(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.lcg128mix_xsl_rr_instance".
        while True:
            yield self.lcg128mix_xsl_rr_instance.random_raw()
    
    
    def __seed_initialize_lcg128mix_dxsm(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The LCG128Mix_DXSM method is initialized with seeds.
        self.lcg128mix_dxsm_instance = LCG128Mix(seed, output = 'dxsm')
    
    
    def __lcg128mix_dxsm(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.lcg128mix_instance".
        while True:
            yield self.lcg128mix_dxsm_instance.random_raw()
    
    
    def __seed_initialize_lcg128mix_murmur3(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The LCG128Mix_MURMUR3 method is initialized with seeds.
        self.lcg128mix_murmur3_instance = LCG128Mix(seed, output = 'murmur3')
    
    
    def __lcg128mix_murmur3(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.lcg128mix_murmur3_instance".
        while True:
            yield self.lcg128mix_murmur3_instance.random_raw()
    
    
    def __seed_initialize_philox_counter(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The PhiloxCounter method is initialized with seeds.
        self.philox_counter_instance = PhiloxCounter(key = seed)
    
    
    def __philox_counter(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.philox_counter_instance".
        while True:
            yield self.philox_counter_instance.random_raw()
    
    
    def __seed_initialize_threefry_counter(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The ThreeFryCounter method is initialized with seeds.
        self.threefry_counter_instance = ThreeFryCounter(key = seed)
    
    
    def __threefry_counter(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.threefry_counter_instance".
        while True:
            yield self.threefry_counter_instance.random_raw()
    
    
    def __seed_initialize_aes_counter(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The AESCounter method is initialized with seeds.
        self.aes_counter_instance = AESCounter(seed, mode = "sequence")
    
    
    def __aes_counter(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.aes_counter_instance".
        while True:
            yield self.aes_counter_instance.random_raw()
    
    
    def __seed_initialize_chacha_counter(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The ChaChaCounter method is initialized with seeds.
        self.chacha_counter_instance = ChaChaCounter(seed, mode = "sequence")
    
    
    def __chacha_counter(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.chacha_counter_instance".
        while True:
            yield self.chacha_counter_instance.random_raw()
    
    
    def __seed_initialize_speck_counter(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The SPECKCounter method is initialized with seeds.
        self.speck_counter_instance = SPECKCounter(seed, mode = "sequence")
    
    
    def __speck_counter(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.speck_counter_instance".
        while True:
            yield self.speck_counter_instance.random_raw()
    
    
    def __seed_initialize_xsm64(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The XSM64 method is initialized with seeds.
        self.xsm64_instance = XSM64(seed)
    
    
    def __xsm64(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.xsm64_instance".
        while True:
            yield self.xsm64_instance.random_raw()
    
    
    def __seed_initialize_efiix64(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The EFIIX64 method is initialized with seeds.
        self.efiix64_instance = EFIIX64(seed)
    
    
    def __efiix64(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.efiix64_instance".
        while True:
            yield self.efiix64_instance.random_raw()
    
    
    def __seed_initialize_splitmix64(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The SplitMix64 method is initialized with seeds.
        self.splitmix64_instance = SplitMix64(seed)
    
    
    def __splitmix64(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.splitmix64_instance".
        while True:
            yield self.splitmix64_instance.random_raw()
    
    
    def __seed_initialize_ran64(self, seed_init_locals: dict, seed: Integer, algorithm_characteristics_parameter: dict) -> None:  #The Ran64 method is initialized with seeds.
        self.ran64_instance = Ran64(seed)
    
    
    def __ran64(self, algorithm_characteristics_parameter: dict) -> Iterator[Integer]:
        #The external variable used is "self.ran64_instance".
        while True:
            yield self.ran64_instance.random_raw()
    
    
    def source_random_number(self) -> Iterator[Integer]:
        '''
            The source random number directly derived from the random generator algorithm.
            
            Iterator
            --------
            source_random_number: Integer
                Returns a pseudo-random number.
            
            Note
            ----
            The value range are determined by the random algorithm, which is specified by parameter 'prng_type' at instance initialization.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> source_random_number = prng_instance.source_random_number()
            >>> next(source_random_number)
            65852230656997158461166665751696465914198450243194923777324019418213544382100
            
            >>> prng_instance = pure_prng(seed, new_prng_period = 2 ** 512)
            >>> source_random_number = prng_instance.source_random_number()
            >>> next(source_random_number)
            8375486648769878807557228126183349922765245383564825377649864304632902242469125910865615742661048315918259479944116325466004411700005484642554244082978452
        '''
        prng_type = self.prng_type
        algorithm_characteristics_parameter = self.__class__.prng_algorithms_dict[prng_type]
        output_width = algorithm_characteristics_parameter['output_width']
        hash_size = algorithm_characteristics_parameter['hash_size']
        hash_callable_dict = self.all_hash_callable_dict[prng_type]
        hash_callable = hash_callable_dict['hash_callable']
        additional_hash_callable = hash_callable_dict['additional_hash_callable']
        while True:
            hash_result = 0
            for i in range(output_width):
                hash_result |= next(hash_callable) << (hash_size * i)
            
            if additional_hash_callable is not None:
                hash_result = additional_hash_callable(hash_result, hash_size)
            
            yield hash_result
    
    
    def rand_bits(self, bit_size: Integer, new_period: Optional[Integer] = None) -> Iterator[Integer]:
        '''
            Get a pseudo random binary number.  得到一个伪随机二进制数。
            
            Parameters
            ----------
            bit_size: Integer
                Sets the pseudo-random number that takes the specified bit length.  设定取指定比特长度的伪随机数。
            
            new_period: Integer, default None
                Set the period of the pseudo-random sequence.
                The default is not to change the eigenperiod of the selected pseudo-random number generator algorithm. 缺省就是不改变所选伪随机数生成算法的本征周期。
            
            Iterator
            --------
            rand_bits: Integer
                Returns a pseudo random number of the specified bit length. 返回一个指定比特长度伪随机数。
            
            Note
            ----
            Set the random number generated after new_period to be slightly unbalanced.  设定new_period后生成的随机数会有很轻微的不均衡。
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> rand_bits = prng_instance.rand_bits(512)
            >>> next(rand_bits)
            mpz(6144768950704661248519702670268583753928668607451020183407159490385670202458730311510261255705698403097105657582435836672179668357656056427608305574891156)
            
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> rand_bits = prng_instance.rand_bits(512, period)
            >>> next(rand_bits)
            mpz(2954964798889411590155032615694646383408546750268072607273800792672971321854983100133610686738061114434885994588970398525439724215184541467422573311905001)
        '''
        assert isinstance(bit_size, (int, type(mpz(0)))), f'bit_size must be an Integer, got type {type(bit_size).__name__}'
        if bit_size <= 0: raise ValueError('bit_size must be > 0')
        
        algorithm_characteristics_parameter = self.__class__.prng_algorithms_dict[self.prng_type]
        surplus_size = 0
        surplus_bit = 0
        output_size = algorithm_characteristics_parameter['output_size']
        
        if new_period is None:
            rand = self.source_random_number()
        else:
            rand = self.rand_with_period(new_period, 'raw_binary_number')
        
        while True:
            result = 0
            if bit_size > surplus_size:
                n = gmpy2_c_div(bit_size - surplus_size, output_size)
                for i in range(n):
                    result |= next(rand) << (output_size * i)
            else:
                n = 0
            
            result <<= surplus_size
            result |= surplus_bit
            surplus_size = n * output_size + surplus_size - bit_size
            surplus_bit = result >> bit_size
            result &= (1 << bit_size) - 1
            yield result
    
    
    def rand_float(self, precision: Optional[Integer] = None, new_period: Optional[Integer] = None) -> Iterator[mpfr]:
        '''
            Generate a pseudo-random real number (you can set the pseudo-random number generator algorithm period).  生成一个伪随机实数（可设定伪随机数生成器算法周期）。
            
            Parameters
            ----------
            precision: Integer, default None
                Output binary floating point precision.  输出的二进制浮点精度。
                Precision must be >= 2
            
            new_period: Integer, default None
                Set the period of the pseudo-random sequence.
                The default is not to change the eigenperiod of the selected pseudo-random number generator algorithm. 缺省就是不改变所选伪随机数生成算法的本征周期。
            
            Iterator
            --------
            rand_float: mpfr
                Returns a pseudo-random real number in [0, 1), with 0 included and 1 excluded.
            
            Note
            ----
            Set the random number generated after new_period to be slightly unbalanced.  设定new_period后生成的随机数会有很轻微的不均衡。
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> rand_float = prng_instance.rand_float(100)
            >>> next(rand_float)
            mpfr('0.56576176351048513846261940831522',100)
        '''
        assert isinstance(precision, (int, type(mpz(0)), type(None))), f'precision must be an Integer or None, got type {type(precision).__name__}'
        
        if precision is None:
            precision = gmpy2_get_context().precision
        else:
            if precision < 2: raise ValueError('precision must be >= 2')
        
        bit_size = precision
        rand_bits = self.rand_bits(bit_size, new_period)
        with gmpy2_local_context(gmpy2_get_context(), precision = precision):
            while True:
                random_number = mpfr(next(rand_bits))
                yield random_number / mpfr(1 << bit_size)
    
    
    def rand_int(self, b: Integer, a: Integer = 0, unbias: bool = True, new_period: Optional[Integer] = None) -> Iterator[Integer]:
        '''
            Generates a pseudo-random integer within a specified interval (can set the pseudo-random number generator algorithm period).  生成一个指定区间内的伪随机整数（可设定伪随机数生成器算法周期）。
            
            Parameters
            ----------
            b: Integer
                Upper bound on the range including 'b'.
            
            a: Integer, default 0
                Lower bound on the range including 'a'.
            
            unbias: bool, default True
                If true, the output pseudo-random number is in unbiased mode.  设真，则输出的伪随机数为非偏置模式。
                But the pseudo-random number output has only the expected period, not the exact fixed period.  但同输出的伪随机数只有期望周期，而不是精确地固定周期。
                
                If false, the output pseudo-random number is biased, but for the exact fixed period.  设假，输出的伪随机数是有偏的，但为精确的固定周期。
            
            new_period: Integer, default None
                Set the period of the pseudo-random sequence.
            
            Iterator
            --------
            rand_int: Integer
                Returns an integer pseudo-random number in the range [a, b].
                
                When the parameter "unbias" is true, the output pseudo-random number period is expected to be 3/4 of 'prng_period'.  当参数“unbias”为真时，输出的伪随机数周期期望为'prng_period'的3/4。
            
            Note
            ----
            The scale from a to b cannot exceed the period of the pseudo-random number generator.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> rand_int = prng_instance.rand_int(100, 1)
            >>> next(rand_int)
            mpz(21)
        '''
        assert isinstance(b, (int, type(mpz(0)))), f'b must be an Integer, got type {type(b).__name__}'
        assert isinstance(a, (int, type(mpz(0)))), f'a must be an Integer, got type {type(a).__name__}'
        assert isinstance(new_period, (int, type(mpz(0)), type(None))), f'new_period must be an Integer or None, got type {type(new_period).__name__}'
        if a > b: raise ValueError('a must be <= b')
        
        difference_value = b - a
        if difference_value == 0:
            while True:
                yield a
        
        difference_bit_size = difference_value.bit_length()
        difference_bit_mask = gmpy2_bit_mask(difference_bit_size)
        
        algorithm_characteristics_parameter = self.__class__.prng_algorithms_dict[self.prng_type]
        prng_period = algorithm_characteristics_parameter['prng_period']
        output_size = algorithm_characteristics_parameter['output_size']
        
        if new_period is None:
            if difference_value >= prng_period: raise ValueError('The a to b difference extends beyond the period of the pseudo-random number generator.')
            if prng_period != float('+inf'):
                random_number_method = self.source_random_number()
            else:
                random_number_method = self.rand_bits(difference_bit_size)
        else:
            if difference_value >= new_period: raise ValueError('The a to b difference extends beyond the period of the pseudo-random number generator.')
            random_number_method = self.rand_with_period(new_period)
        
        if unbias:
            while True:
                while not ((random_number:= next(random_number_method) & difference_bit_mask) <= difference_value): pass
                yield a + random_number
        else:
            scale = difference_value + 1
            while True:
                random_number = (next(random_number_method) * scale) >> output_size
                yield a + random_number
    
    
    def get_randint_set(self, b: Integer, a: Integer, k: Integer) -> Iterator[Set[Integer]]:
        '''
            Generates a set of pseudo-random integers in a specified interval. 生成一个指定区间内的伪随机整数的集合。
            
            Parameters
            ----------
            b: Integer
                Upper bound on the range including 'b'.
            
            a: Integer
                Lower bound on the range including 'a'.
            
            k: Integer
                The number of set elements to generate.
            
            Iterator
            --------
            get_randint_set: Set[Integer]
                Returns a set of pseudo-random Numbers with k elements in the range [a, b].
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> get_randint_set = prng_instance.get_randint_set(100, 1, 6)
            >>> next(get_randint_set)
            {mpz(34), mpz(99), mpz(37), mpz(45), mpz(19), mpz(21)}
        '''
        assert isinstance(b, (int, type(mpz(0)))), f'b must be an Integer, got type {type(b).__name__}'
        assert isinstance(a, (int, type(mpz(0)))), f'a must be an Integer, got type {type(a).__name__}'
        assert isinstance(k, (int, type(mpz(0)))), f'k must be an Integer, got type {type(k).__name__}'
        if a > b: raise ValueError('a must be <= b')
        if k < 0: raise ValueError('k must be >= 0')
        if k > (b - a + 1): raise ValueError("k must be <= b - a + 1")
        
        rand_int = self.rand_int(b, a)
        while True:
            randint_set: Set[Integer] = set()
            while len(randint_set) < k:
                randint_set.add(next(rand_int))
            yield randint_set
    
    
    def rand_with_period(self, new_period: Integer, output_value_type: str = 'randomized_consecutive_numbers') -> Iterator[Integer]:
        '''
            Generates an integer pseudo-random number with a specified period.
            
            Parameters
            ----------
            new_period: Integer
                Set the period of the pseudo-random sequence.
            
            output_value_type: str['randomized_consecutive_numbers' | 'raw_binary_number'], default 'randomized_consecutive_numbers'
                Returns a randomized continuous number with the parameter 'randomized_consecutive_numbers'.  指定返回随机化连续数或全长度二进制随机数。
                Returns a full-length binary random number with the parameter 'raw_binary_number'.
            
            Iterator
            --------
            rand_with_period: Integer
                Returns a randomized continuous number or a full length binary random number for a new period.  返回一个随机化连续数或新周期下的全长度二进制随机数。
                The value range for full-length binary random Numbers is [0, Can fully express the binary length of the PRNG eigenperiod used)
                The value range of a randomized continuous number is [0, new_period), with zero included and new_period excluded.
            
            Note
            ----
            Set new_period to be no less than 2.
            
            The value of (new period / original period) is the representation of generating efficiency.
            When the difference between the new period and the original period is too large, it may takes a long time to generate a pseudo-random number!
            
            When the pseudo-random algorithm is infinite period, this change period method cannot be used.
            
            Examples
            --------
            >>> seed = 170141183460469231731687303715884105727
            >>> prng_instance = pure_prng(seed)
            >>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
            >>> rand_with_period = prng_instance.rand_with_period(period)
            >>> next(rand_with_period)
            mpz(65852230656997158461166665751696465914198450243194923777324019418213544381986)
            
            >>> rand_with_period = prng_instance.rand_with_period(period, 'raw_binary_number')
            >>> next(rand_with_period)
            mpz(53067260390396280968027884646874354062063398901623645439544105836818444733296)
        '''
        assert isinstance(new_period, (int, type(mpz(0)))), f'new_period must be an Integer, got type {type(new_period).__name__}'
        if new_period < 2: raise ValueError('new_period must be >= 2')
        prng_period = self.__class__.prng_algorithms_dict[self.prng_type]['prng_period']
        if prng_period == float('+inf'): raise ValueError('The current pseudo-random algorithm is infinite period, and this change period method cannot be used.')
        if new_period > prng_period: raise ValueError('Suppose the new period number cannot be greater than the original period number of the pseudorandom number generator.')
        if output_value_type not in ('randomized_consecutive_numbers', 'raw_binary_number'): raise ValueError('')
        
        temporary_prng_type = self.prng_type
        self.prng_type = 'internal_generator'  #Switch to a separate pseudo-random sequence inside.
        get_randint_set = self.get_randint_set(prng_period - 1, 0, prng_period - new_period)
        next(get_randint_set)
        rand_bits = self.rand_bits(8)
        next(rand_bits)
        self.prng_type = temporary_prng_type
        
        source_random_number = self.source_random_number()
        
        update_count = 0
        while True:
            if update_count == 0:  #Update the rules for the cache.  更新缓存的规则。
                set_of_numbers_to_exclude = next(get_randint_set)
                ordered_list_of_excluded_numbers = sorted(set_of_numbers_to_exclude)
                update_count = next(rand_bits)
            else:
                update_count -= 1
            
            quotient, random_number = gmpy2_t_divmod(next(source_random_number), prng_period)
            while True:
                if random_number not in set_of_numbers_to_exclude:
                    break
                quotient, random_number = gmpy2_t_divmod(next(source_random_number), prng_period)
            
            if output_value_type == 'randomized_consecutive_numbers':
                randomized_consecutive_numbers = random_number - bisect_left(ordered_list_of_excluded_numbers, random_number)
                yield randomized_consecutive_numbers
            elif output_value_type == 'raw_binary_number':
                random_number += quotient * prng_period
                yield random_number
