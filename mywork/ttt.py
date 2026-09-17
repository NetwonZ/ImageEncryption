from .Encryption import Encrypter
from .SalomonCouplingCML import SalomoncouplingCML 
from .KeyStream import keystream_generation_v3, keystream_generation, DEFAULT_USER_KEY_BITS_V3, DEFAULT_USER_KEY_BITS
import numpy as np

L = 512*512
user_key = DEFAULT_USER_KEY_BITS_V3
key = keystream_generation_v3(L, "C:\\ImageEncryption\\images\\img3.png", user_key)

params = {
    "mu": float(key["mu"]),
    "v": float(key["v"]),
    "alpha": float(key["alpha"]),
    "beta": float(key["beta"]),
    "xi": 1,
    "eta": 1,
}
x0 = np.array(key["X"])

# params = {
#     "mu": 8.0,
#     "v": 7.0,
#     "alpha": 7.0,
#     "beta": 4.0,
#     "xi": 1,
#     "eta": 1,
# }
# seed = 2026
# np.random.seed(seed)
# x0 = np.random.rand(L)



cml = SalomoncouplingCML(L=L, params=params, initstate={"x0": x0})
e = Encrypter(cml=cml, config=None, global_permutation_method=None, user_key=None)

ei = e.encrypt(image= "C:\\ImageEncryption\\images\\img3.png")
ei = ei.encrypted_image

import matplotlib.pyplot as plt

plt.imshow(ei)
plt.show()