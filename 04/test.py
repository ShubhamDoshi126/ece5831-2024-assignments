import numpy as np
import matplotlib.pyplot as plt

# %%
# step_function numpy version
def step_function(x):
    return (x > 0).astype(int)


# %%
x = np.arange(-10.0, 10.0, 0.1)
y = step_function(x)

# %%
plt.plot(x, y)
# you have to call `show` to see the graph in command line interface.
plt.show()