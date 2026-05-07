import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("vertical-high-angle-shot-highway-surrounded-by-trees-cloudy-grey-sky.jpg")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges
edges = cv2.Canny(blur, 50, 150)

# Create mask
height = image.shape[0]
polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
])

mask = np.zeros_like(edges)

cv2.fillPoly(mask, polygons, 255)

masked_image = cv2.bitwise_and(edges, mask)

# Detect lines
lines = cv2.HoughLinesP(
    masked_image,
    2,
    np.pi / 180,
    100,
    np.array([]),
    minLineLength=40,
    maxLineGap=5
)

# Draw lines
line_image = np.zeros_like(image)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# Combine images
combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

# Show output
plt.imshow(combo_image)
plt.title("Lane Detection")
plt.axis("off")
plt.show()
