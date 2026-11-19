from fractions import Fraction
ARB_PRECISIONS = (128, 256, 512, 1024, 2048, 4096)
XCHECK_EXTRA_BITS = 64
KL_BINS = (('B_c', Fraction(1, 10), Fraction(1)),) + tuple(((f'B_{k}', Fraction(1, 10 ** (k + 1)), Fraction(1, 10 ** k)) for k in range(1, 15)))
CERTIFICATION_FAILURE_LIMIT = Fraction(1, 1000)
