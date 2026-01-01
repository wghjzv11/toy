from scipy.stats import norm

import math
import pandas as pd
import matplotlib.pyplot as plt

def get_price(S: float,
              K: float,
              r: float,
              T: float,
              vol: float,
              option: str = "call",
              q: float = 0.0) -> float:
    """
    S: stock price
    K: strike price
    r: risk free interest rate
    T: time to maturity
    vol: volatility
    q: divdend rate
    """
    d1 = ((math.log(S / K) + (r - q + 0.5 * vol ** 2) * T)) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    if option=="call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def get_implied_vol(price: float, 
                S: float,
                K: float,
                r: float,
                T: float,
                option: str = "call",
                q: float = 0.0,
                vol_low: float = 1e-8,
                vol_high: float = 5.0,
                tol: float = 1e-10,
                max_iter: int = 200):
    """
    S: stock price
    K: strike price
    r: risk free interest rate
    T: time to maturity
    option: type of option ("call"|"put")
    q: divdend rate
	vol_low: minimum vol(default=1e-8)
	vol_high: maximum vol(default=5.0)
	tol: tolerance(default=1e-10)
	max_iter: maximum iteration(default=200)
    """
    def intrinsic_value(S, K, r, T, q, option):
        disc = math.exp(-r * T)
    
        if option == "call":
            return max(S - K*disc, 0.0)
        else:
            return max(K*disc - S, 0.0)
    
    def f(vol: float) -> float: # 최소화해야 하는 함수
        return get_price(S, K, r, T, vol, option, q) - price

    if abs(price - intrinsic_value(S, K, r, T, q, option)) <=  1e-2:
        return 0.0
        
    lo, hi = vol_low, vol_high
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(f"지정한 변동성 구간[{vol_low}, {vol_high}]에서 해를 찾을 수 없습니다.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid

        if flo * fmid <= 0: # 변동성이 flo와 fmid 사이임
            hi, fhi = mid, fmid
        else:               # 변동성이 mid와 fmid 사이임
            lo, flo = mid, fmid

    return 0.5 * (lo + hi)