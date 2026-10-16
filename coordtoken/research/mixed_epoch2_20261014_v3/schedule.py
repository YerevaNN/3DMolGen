import math

def lr_factor(samples,rows):
    if not 0<=samples<=rows:raise ValueError('Sample count outside epoch')
    if samples<=.8*rows:return 1.
    return .5*(1+math.cos(math.pi*(samples-.8*rows)/(.2*rows)))
