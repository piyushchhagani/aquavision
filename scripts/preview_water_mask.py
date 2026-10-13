from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from aquavision.water.mask import segment_water_hsv, calculate_water_coverage
p = Path("plastic_test.jpg")
image = cv2.imread(str(p))
if image is None:
    raise FileNotFoundError(p)
mask = segment_water_hsv(image)
print(f"Experimental mask coverage: {calculate_water_coverage(mask):.2f}%")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); ax[0].set_title("Original")
ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Experimental HSV mask")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
