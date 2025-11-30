import numpy as np
import matplotlib.pyplot as plt

cm = np.load("pytorch-training-workflow_at2/outputs/confusion_matrix.npy")

plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.savefig("confusion_matrix.png")   # <-- Save to file
