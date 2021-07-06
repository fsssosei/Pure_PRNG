# Pure_PRNG

![PyPI](https://img.shields.io/pypi/v/pure_prng?color=red)
![PyPI - Status](https://img.shields.io/pypi/status/pure_prng)
![GitHub Release Date](https://img.shields.io/github/release-date/fsssosei/pure_prng)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/fsssosei/Pure_PRNG.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/fsssosei/Pure_PRNG/context:python)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bf34f8d12be84b4492a5a3709df0aae5)](https://www.codacy.com/manual/fsssosei/pure_prng?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=fsssosei/pure_prng&amp;utm_campaign=Badge_Grade)
![PyPI - Downloads](https://img.shields.io/pypi/dw/pure_prng?label=PyPI%20-%20Downloads)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pure_prng)
![PyPI - License](https://img.shields.io/pypi/l/pure_prng)

*Generate professional pseudo-random number package in python.*

Only the pseudo-random number algorithm with good statistical properties is implemented.

There are "methods" that specify the period of a multi-precision pseudo-random sequence.

## Realized pseudo-random number generation algorithm

Quadratic Congruential Generator

Cubic Congruential Generator

Inversive Congruential Generator

PCG64_XSL_RR; PCG64_DXSM

LCG64_32_ext

LCG128Mix_XSL_RR; LCG128Mix_DXSM; LCG128Mix_MURMUR3

XSM64

EFIIX64

PhiloxCounter

ThreeFryCounter

AESCounter

ChaChaCounter

SPECKCounter

SquaresCounter

## Installation

Installation can be done through pip. You must have python version >= 3.8

	pip install pure-prng

## Usage

The statement to import the package:

	from pure_prng_package import pure_prng

Example:

	>>> seed = 170141183460469231731687303715884105727
	>>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
	
	>>> prng_instance = pure_prng(seed)
	>>> source_random_number = prng_instance.source_random_number()
	>>> next(source_random_number)
	65852230656997158461166665751696465914198450243194923777324019418213544382100
	
	>>> prng_instance = pure_prng(seed, new_prng_period = 2 ** 512)
	>>> source_random_number = prng_instance.source_random_number()
	>>> next(source_random_number)
	8375486648769878807557228126183349922765245383564825377649864304632902242469125910865615742661048315918259479944116325466004411700005484642554244082978452
	
	>>> prng_instance = pure_prng(seed)
	>>> rand_bits = prng_instance.rand_bits(512)
	>>> next(rand_bits)
	mpz(6144768950704661248519702670268583753928668607451020183407159490385670202458730311510261255705698403097105657582435836672179668357656056427608305574891156)
	>>> rand_bits = prng_instance.rand_bits(512, period)
	>>> next(rand_bits)
	mpz(2954964798889411590155032615694646383408546750268072607273800792672971321854983100133610686738061114434885994588970398525439724215184541467422573311905001)
	
	>>> prng_instance = pure_prng(seed)
	>>> rand_float = prng_instance.rand_float(100)
	>>> next(rand_float)
	mpfr('0.56576176351048513846261940831522',100)
	
	>>> prng_instance = pure_prng(seed)
	>>> rand_int = prng_instance.rand_int(100, 1)
	>>> next(rand_int)
	mpz(21)
	
	>>> prng_instance = pure_prng(seed)
	>>> get_randint_set = prng_instance.get_randint_set(100, 1, 6)
	>>> next(get_randint_set)
	{mpz(34), mpz(99), mpz(37), mpz(45), mpz(19), mpz(21)}
	
	>>> prng_instance = pure_prng(seed)
	>>> rand_with_period = prng_instance.rand_with_period(period)
	>>> next(rand_with_period)
	mpz(65852230656997158461166665751696465914198450243194923777324019418213544381986)
	>>> rand_with_period = prng_instance.rand_with_period(period, 'raw_binary_number')
	>>> next(rand_with_period)
	mpz(53067260390396280968027884646874354062063398901623645439544105836818444733296)

##Future work

The following algorithm is intended to be implemented:

MIXMAX

NLFSR

Salmon

GPU Philox
