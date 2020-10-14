# pure_prng

![PyPI](https://img.shields.io/pypi/v/pure_prng?color=red)
![PyPI - Status](https://img.shields.io/pypi/status/pure_prng)
![GitHub Release Date](https://img.shields.io/github/release-date/fsssosei/pure_prng)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/fsssosei/Pure_PRNG.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/fsssosei/Pure_PRNG/context:python)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bf34f8d12be84b4492a5a3709df0aae5)](https://www.codacy.com/manual/fsssosei/pure_prng?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=fsssosei/pure_prng&amp;utm_campaign=Badge_Grade)
![PyPI - Downloads](https://img.shields.io/pypi/dw/pure_prng?label=PyPI%20-%20Downloads)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pure_prng)
![PyPI - License](https://img.shields.io/pypi/l/pure_prng)

*Generate multi-precision pseudo-random number package in python.*

Xoshiro256++ algorithm has been completed.

There are "methods" that specify the period of a multi-precision pseudo-random sequence.

## Installation

Installation can be done through pip. You must have python version >= 3.7

	pip install pure-prng

## Usage

The statement to import the package:

	from pure_prng_package import pure_prng
	
Example:

	>>> seed = 170141183460469231731687303715884105727
	>>> period = 115792089237316195423570985008687907853269984665640564039457584007913129639747
	
	>>> prng_instance = pure_prng(seed)
	
	>>> prng_instance.source_random_number()
	73260932800743358445652462028207907455677987852735468159219395093090100006110
	
	>>> prng_instance.rand_float()
	mpfr('0.6326937641706669741872583730940429737405414921354622618051716414693676562568086',257)
	>>> prng_instance.rand_float(period)
	mpfr('0.02795744845257346733436109648463446736744766610965612207643215290679786849298934',256)
	
	>>> prng_instance.rand_int(100, 1)
	mpz(94)
	>>> prng_instance.rand_int(100, 1, period)
	mpz(38)
	
	>>> prng_instance.get_randint_set(100, 1, 6)
	{mpz(98), mpz(68), mpz(46), mpz(24), mpz(27), mpz(94)}
	
	>>> prng_instance.rand_with_period(period)
	40688839126177430252467309162469901643963863918059158449302074429100738061310
