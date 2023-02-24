"""
Works.
Image Segmentation.md
"""
import cv2

# Load the image
# img = cv2.imread('image.jpg')
img = cv2.imread('colored-dots.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold to create a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw a circle around each contour
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    # cv2.circle(img, center, radius, (0, 255, 0), 2)  # Lime
    cv2.circle(img, center, radius, (0, 0, 0), 2)  # Black

# Display the resulting image
cv2.imshow('Result', img)
cv2.waitKey(0)
