#%%
import matplotlib.pyplot as plt


def schedule(epoch, lr):
    #TODO compelete the scheduler
    if epoch < 640:
        lr = 0.001
    else:
        lr = 0.0001
    return lr

x = [_ for _ in range(5000)]
y = [schedule(_, 0) for _ in x]

plt.plot(x, y)
plt.show()
