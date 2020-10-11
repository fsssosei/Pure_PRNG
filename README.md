# pseudo_random_number_generator

![PyPI](https://img.shields.io/pypi/v/pseudo-random-number-generator?color=red)
![PyPI - Status](https://img.shields.io/pypi/status/pseudo-random-number-generator)
![GitHub Release Date](https://img.shields.io/github/release-date/fsssosei/pseudo_random_number_generator)
[![Build Status](https://scrutinizer-ci.com/g/fsssosei/pseudo_random_number_generator/badges/build.png?b=main)](https://scrutinizer-ci.com/g/fsssosei/pseudo_random_number_generator/build-status/main)
[![Code Intelligence Status](https://scrutinizer-ci.com/g/fsssosei/pseudo_random_number_generator/badges/code-intelligence.svg?b=main)](https://scrutinizer-ci.com/code-intelligence)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/fsssosei/pseudo_random_number_generator.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/fsssosei/pseudo_random_number_generator/context:python)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bf34f8d12be84b4492a5a3709df0aae5)](https://www.codacy.com/manual/fsssosei/pseudo_random_number_generator?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=fsssosei/pseudo_random_number_generator&amp;utm_campaign=Badge_Grade)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/fsssosei/pseudo_random_number_generator/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/fsssosei/pseudo_random_number_generator/?branch=main)
![PyPI - Downloads](https://img.shields.io/pypi/dw/pseudo-random-number-generator?label=PyPI%20-%20Downloads)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pseudo-random-number-generator)
![PyPI - License](https://img.shields.io/pypi/l/pseudo-random-number-generator)

*Generate pseudo-random number package in python.*

Xoshiro256++ algorithm has been completed.

There are "methods" that specify the period of a pseudo-random sequence.

## Installation

Installation can be done through pip. You must have python version >= 3.7

	pip install pseudo-random-number-generator

## Usage

The statement to import the package:

	from pseudo_random_number_generator_package import prng_class
	
Example:

	>>> prng_instance = prng_class(170141183460469231731687303715884105727)
	
	>>> prng_instance.source_random_number()
	73260932800743358445652462028207907455677987852735468159219395093090100006110
	
	>>> prng_instance.rand_float()
	0.6326937641706669741872583730940429737405414921354622618051716414693676562568173
	>>> prng_instance.rand_float(115792089237316195423570985008687907853269984665640564039457584007913129639747)
	0.3513956730047900426657412504296283135563682862601256213620310087617832883445811
	
	>>> prng_instance.rand_int(100, 1)
	64
	>>> prng_instance.rand_int(100, 1, 115792089237316195423570985008687907853269984665640564039457584007913129639747)
	36
	
	>>> prng_instance.generate_set_of_integer_random_numbers(100, 1, 6)
	{64, 39, 9, 41, 23, 92}
	
	>>> prng_instance.random_integer_number_with_definite_period(115792089237316195423570985008687907853269984665640564039457584007913129639747)
	73260932800743358445652462028207907455677987852735468159219395093090100006110
