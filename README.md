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

Xoshiro256++ algorithm has been completed.

There are "methods" that specify the period of a multi-precision pseudo-random sequence.

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
	
	>>> prng_instance.source_random_number()
	63704397730169193686456860639078459647664747236380824242857347684562650854070
	
	>>> prng_instance.rand_float()
	mpfr('0.5501619164985171033237722626311247973459654767593884512471329918122925574973017',257)
	>>> prng_instance.rand_float(period)
	mpfr('0.6665079772632617788674079157248027245703466196226109430388828957294865649611888',256)
	
	>>> prng_instance.rand_int(100, 1)
	mpz(54)
	>>> prng_instance.rand_int(100, 1, period)
	mpz(61)
	
	>>> prng_instance.get_randint_set(100, 1, 6)
	{mpz(5), mpz(9), mpz(18), mpz(54), mpz(57), mpz(93)}
	
	>>> prng_instance.rand_with_period(period)
	103085265137502064472166298218885841110988755115459404830932952476483720814169
