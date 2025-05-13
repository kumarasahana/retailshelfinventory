import cv2
import numpy as np

# Read the image
image = cv2.imread("retail.jpg")
output = image.copy()

# Step 1: Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Step 2: Detect shelf regions using lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Step 3: Detect product contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

product_count = 0
empty_count = 0

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 30 and h > 30:
        roi = gray[y:y+h, x:x+w]
        mean_intensity = roi.mean()
        
        # Decide if empty or product
        if mean_intensity > 200:
            cv2.putText(output, "Empty", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            empty_count += 1
        else:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            product_count += 1

# Step 4: Display counts on screen
cv2.putText(output, f"Products Detected: {product_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(output, f"Empty Spaces: {empty_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Shelf Inventory Monitoring", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
