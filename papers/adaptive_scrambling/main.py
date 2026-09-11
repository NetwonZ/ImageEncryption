import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # papers/ 目录
from performance_analysis import histogram_variance
from .encryption import encrypt
import numpy as np

testimages = [
    r"C:\ImageEncryption\images\img1.png",
    r"C:\ImageEncryption\images\img2.png",
    r"C:\ImageEncryption\images\img3.png",
    r"C:\ImageEncryption\images\img4.png",
    r"C:\ImageEncryption\images\img5.png",
    # r"C:\ImageEncryption\images\img6.png",
    # r"C:\ImageEncryption\images\img7.png",
    # r"C:\ImageEncryption\images\img8.png"
]
hv = []
hv_overall = []
for imgpth in testimages:
    # pimghv = histogram_variance(imgpth)
    # print(f"plain image{imgpth},with HV:{pimghv}")
    for i in range(5):
        secret = f"202600{i}"
        epimg = encrypt(image=imgpth, secret=secret)
        epimg = epimg[0]
        # from matplotlib import pyplot as plt
        # plt.imshow(epimg)
        # plt.show()
        hv_t = histogram_variance(epimg)
        hv.append(hv_t)
    avg_hv = np.mean(hv)
    hv_overall.append(avg_hv)
    print(f"image{imgpth},with HV:{avg_hv}")
print(f"overall HV:{np.mean(hv_overall)}")
    