import pandas as pd
import numpy as np
logs = pd.DataFrame.from_csv(path='VAE_10db_1_6/vae_10db_1_6.log')

#%%
import matplotlib.pyplot as plt

psnrs = logs['val_PSNR'].to_numpy()[2000:]
epchs = np.array([_ for _ in range(psnrs.shape[0])])


f = np.polyfit(epchs, psnrs, 1)
print(f)
plt.plot(epchs, psnrs)
plt.plot(epchs, epchs*f[0]+f[1])
plt.show()
#%%
logs2 = pd.DataFrame.from_csv(path='VAE_10db_1_6/vae_10db_1_6.log')

#%%
psnrs2 = logs2['val_PSNR'].to_numpy()
epchs = np.array([_ for _ in range(psnrs.shape[0])])
epchs = np.concatenate((epchs, [_+2242 for _ in range(psnrs2.shape[0])]), axis=0)[2500:]
psnrs_all = np.concatenate((psnrs, psnrs2), axis=0)[2500:]
f = np.polyfit(epchs, psnrs_all, 1)
print(f)
plt.plot(epchs, psnrs_all)
plt.plot(epchs, epchs*f[0]+f[1])
plt.show()
