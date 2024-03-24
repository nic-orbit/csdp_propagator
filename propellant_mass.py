import math as math
import numpy as np

# code by Niek

n = 3 # manoeuvres needed
mu = 398600.4
mend = 40 #fixed dry mass
mvar = 0.2 * n
Isp = 250
Dv_man = 100
a = (2*6378 + 200 +550)/2
mprop = 0

for j in range(10):
    mprop = mend* math.e**(Dv_man/(Isp *9.81)) - mend
    mend = mprop + mend
    print(j + 1, mend)