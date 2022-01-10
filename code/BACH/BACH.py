import pymc3 as pm
import theano.tensor as T
from theano.compile.ops import as_op
import numpy as np

@as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dscalar])
def cooling_history_1(start_age, start_temp, rate_1, end_age):
#     assume we start from the oldest 
    end_temp = start_temp - (start_age - end_age)*rate_1
    return np.array(end_temp)

def get_cooling_history_1(start_age, start_temp, rate_1, end_age):
#     assume we start from the oldest 
    end_temp = start_temp - (start_age - end_age)*rate_1
    return np.array(end_temp)

@as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dscalar])
def cooling_history_2(start_age, start_temp, rate_1, rate_2, changepoint_1, changepoint_2, end_age):
#     assume we start from the oldest 
    this_temp = start_temp
    this_age = start_age
    end_temp = None
    for changepoint, rate in sorted(zip([changepoint_1, changepoint_2], [rate_1, rate_2]),reverse=1):
        if end_age > changepoint:
            this_temp =  this_temp - (this_age-end_age) * rate
            break
        else:
            this_temp = this_temp - (this_age-changepoint) * rate
            this_age = changepoint
    return np.array(this_temp)

def get_cooling_history_2(start_age, start_temp, rate_1, rate_2, changepoint_1, changepoint_2, end_age):
#     assume we start from the oldest 
    this_temp = start_temp
    this_age = start_age
    end_temp = None
    for changepoint, rate in sorted(zip([changepoint_1, changepoint_2], [rate_1, rate_2]),reverse=1):
        if end_age > changepoint:
            this_temp =  this_temp - (this_age-end_age) * rate
            break
        else:
            this_temp = this_temp - (this_age-changepoint) * rate
            this_age = changepoint
    return this_temp

@as_op(itypes=[T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dscalar])
def cooling_history_3(start_age, start_temp, rate_1, rate_2, rate_3, changepoint_1, changepoint_2, changepoint_3, end_age):
#     assume we start from the oldest 
    this_temp = start_temp
    this_age = start_age
    end_temp = None
    for changepoint, rate in sorted(zip([changepoint_1, changepoint_2, changepoint_3], [rate_1, rate_2, rate_3]),reverse=1):
        if end_age > changepoint:
            this_temp =  this_temp - (this_age-end_age) * rate
            break
        else:
            this_temp = this_temp - (this_age-changepoint) * rate
            this_age = changepoint
    return np.array(this_temp)

def get_cooling_history_3(start_age, start_temp, rate_1, rate_2, rate_3, changepoint_1, changepoint_2, changepoint_3, end_age):
#     assume we start from the oldest 
    this_temp = start_temp
    this_age = start_age
    end_temp = None
    for changepoint, rate in sorted(zip([changepoint_1, changepoint_2, changepoint_3], [rate_1, rate_2, rate_3]),reverse=1):
        if end_age > changepoint:
            this_temp =  this_temp - (this_age-end_age) * rate
            break
        else:
            this_temp = this_temp - (this_age-changepoint) * rate
            this_age = changepoint
    return this_temp

