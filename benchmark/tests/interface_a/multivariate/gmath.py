from __future__ import annotations
from decimal import Decimal, localcontext, getcontext
from fractions import Fraction as Fr
from functools import lru_cache
EULER = Decimal('0.57721566490153286060651209008240243104215933593992359880576723488486772677766467093694706329174674951463144724980708248096050401448654283622417399764492353625350033374293733773767394279259525824709491600873520394816567085323315177661152862119950150798479374508570574002992135478614669402960432542151905877553526733139925401296742051375413954911168510280798423487758720503843109399736137255306088933126760017247953783675927135157722610273492913940798430103417771778088154957066107501016191663340152278935867965497252036212879226555953669628176388')

@lru_cache(None)
def bernoulli(n):
    A = [Fr(0)] * (n + 1)
    for m in range(n + 1):
        A[m] = Fr(1, m + 1)
        for j in range(m, 0, -1):
            A[j - 1] = j * (A[j - 1] - A[j])
    return A[0] if n != 1 else Fr(-1, 2)

def _D(fr):
    return Decimal(fr.numerator) / Decimal(fr.denominator)

def zeta(s, prec=80):
    with localcontext() as c:
        c.prec = prec + 20
        N = 40
        S = Decimal(s)
        tot = sum((Decimal(n) ** (-S) for n in range(1, N)))
        tot += Decimal(N) ** (1 - S) / (S - 1) + Decimal(N) ** (-S) / 2
        fac = Decimal(1)
        rising = S
        for j in range(1, 40):
            term = _D(bernoulli(2 * j)) / _factorial(2 * j) * rising * Decimal(N) ** (-S - 2 * j + 1)
            tot += term
            rising *= (S + 2 * j - 1) * (S + 2 * j)
            if abs(term) < Decimal(10) ** (-(prec + 10)):
                break
        return +tot

@lru_cache(None)
def _factorial(n):
    r = Decimal(1)
    for k in range(2, n + 1):
        r *= k
    return r

def _shift(x, Y0):
    n = 0
    while x + n < Y0:
        n += 1
    return n

def lgamma(x, prec):
    with localcontext() as c:
        c.prec = prec + 20
        x = Decimal(x)
        Y0 = Decimal(prec)
        n = _shift(x, Y0)
        y = x + n
        s = (y - Decimal('0.5')) * y.ln() - y + (2 * Decimal(_pi(prec + 20))).ln() / 2
        yk = y
        y2 = y * y
        for k in range(1, 200):
            term = _D(bernoulli(2 * k)) / (2 * k * (2 * k - 1)) / yk
            s += term
            yk *= y2
            if abs(term) < Decimal(10) ** (-(prec + 15)):
                break
        prod_log = sum(((x + j).ln() for j in range(n))) if n else Decimal(0)
        return +(s - prod_log)

def digamma(x, prec):
    with localcontext() as c:
        c.prec = prec + 20
        x = Decimal(x)
        Y0 = Decimal(prec)
        n = _shift(x, Y0)
        y = x + n
        s = y.ln() - 1 / (2 * y)
        y2 = y * y
        yk = y2
        for k in range(1, 200):
            term = _D(bernoulli(2 * k)) / (2 * k) / yk
            s -= term
            yk *= y2
            if abs(term) < Decimal(10) ** (-(prec + 15)):
                break
        s -= sum((1 / (x + j) for j in range(n))) if n else Decimal(0)
        return +s

@lru_cache(None)
def _pi(prec):
    with localcontext() as c:
        c.prec = prec + 10

        def arctan_inv(n):
            x = Decimal(1) / n
            x2 = x * x
            s = x
            t = x
            k = 1
            while True:
                t *= -x2
                k += 2
                term = t / k
                if abs(term) < Decimal(10) ** (-(prec + 8)):
                    break
                s += term
            return s
        return +(16 * arctan_inv(5) - 4 * arctan_inv(239))
