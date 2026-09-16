from std.math import exp, log, erf, erfc, sin, sqrt


def main():
    # Correctly rounded float64 values, from mpmath at 50 digits.
    print("exp(1.0)   =", exp(Float64(1.0)), " true 2.718281828459045")
    print("log(0.02)  =", log(Float64(0.02)), " true -3.912023005428146")
    print("erf(0.5)   =", erf(Float64(0.5)), " true 0.5204998778130465")
    print("sin(1.0)   =", sin(Float64(1.0)), " true 0.8414709848078965")
    print("sqrt(2.0)  =", sqrt(Float64(2.0)), " true 1.4142135623730951")
    print("erfc(0.5)  =", erfc(Float64(0.5)), " true 0.4795001221869535")
