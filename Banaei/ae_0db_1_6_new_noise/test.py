import matplotlib.pyplot as plt
import pandas as pd
logs = pd.DataFrame.from_csv(path='Banaei/ae_0db_1_6_new_noise/ae_0db_1_6_nnoise_cont3.log')



#%%
print(logs)

#%%
import numpy as np
psnrs = logs['val_PSNR'].to_numpy()
epchs = np.array([_ for _ in range(psnrs.shape[0])])
#%%


def plot_psnrs(psnrs, epchs):
    f = np.polyfit(epchs, psnrs, 1)
    print(f)
    plt.plot(epchs, psnrs)
    plt.plot(epchs, epchs * f[0] + f[1])
    plt.show()

#%%
plot_psnrs(psnrs, epchs)