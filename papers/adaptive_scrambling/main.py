import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # papers/ 目录
from performance_analysis import histogram_variance
from .encryption import encrypt

testimages = [
    r"C:\ImageEncryption\images\img1.png",
    r"C:\ImageEncryption\images\img2.png",
    r"C:\ImageEncryption\images\img3.png",
    r"C:\ImageEncryption\images\img4.png",
    r"C:\ImageEncryption\images\img5.png",
    r"C:\ImageEncryption\images\img6.png",
    r"C:\ImageEncryption\images\img7.png",
    r"C:\ImageEncryption\images\img8.png"
]

for imgpth in testimages:
    epimg = encrypt(image=imgpth)
    epimg = epimg[0]
    # from matplotlib import pyplot as plt
    # plt.imshow(epimg)
    # plt.show()
    hv = histogram_variance(epimg)
    print(f"Image:{imgpth},with{hv}")
    
    