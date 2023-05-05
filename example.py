import time
import numpy as np
import matplotlib.pyplot as pl
from math import exp
from math import log
from math import sqrt
from scipy.stats import norm
import mocaxpy

def call_option(spot, strike, time_to_maturity, volatility, risk_free_rate):
    nominator = log(spot / strike) + (risk_free_rate + 0.5 * (volatility * volatility)) * time_to_maturity
    denominator = volatility * sqrt(time_to_maturity)
    d1 = nominator / denominator
    d2 = d1 - denominator
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    df = exp(-risk_free_rate * time_to_maturity)
    value = spot * n_d1 - strike * df * n_d2
    return value

# full revaluation
lowest_spot = 50
highest_spot = 150
n_spots = 1000000
spots = list(np.linspace(lowest_spot, highest_spot, n_spots))
start_timer = time.perf_counter()
call_options_full_revaluation = [call_option(spot, 100.0, 1.0, 0.25, 0.01) for spot in spots]
end_timer = time.perf_counter()
print('Full revaluation {} seconds'.format(end_timer - start_timer))

# approximation by using MoCaX
n_dimension = 1
n_chebyshev_points = 25
domain = mocaxpy.MocaxDomain([[lowest_spot, highest_spot]])
accuracy = mocaxpy.MocaxNs([n_chebyshev_points])
mocax = mocaxpy.Mocax(None, n_dimension, domain, None, accuracy)
chebyshev_points = mocax.get_evaluation_points()
start_timer = time.perf_counter()
option_values_at_evaluation_points = [call_option(point[0], 100.0, 1.0, 0.25, 0.01) for point in chebyshev_points]
mocax.set_original_function_values(option_values_at_evaluation_points)
call_options_approximations = [mocax.eval([spot], derivativeId=0) for spot in spots]
end_timer = time.perf_counter()
print('MoCax approximation {} seconds'.format(end_timer - start_timer))

error_function = np.array(call_options_full_revaluation) - np.array(call_options_approximations)
print('Maximum error {}'.format(np.max(error_function)))

pl.plot(spots, error_function, label='error function', color='blue')
pl.show()
