from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
args = vars(ap.parse_args())

##PREPROCESS
# load the image
image = cv2.imread(args["image"])
print(f"Image loaded: {image.shape}")

# resize image for easier processing
image = imutils.resize(image, width=1000)
print(f"Resized image: {image.shape}")

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

# apply bilateral filter to reduce noise while preserving edges
bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
cv2.imshow("Bilateral Filter", bilateral)
cv2.waitKey(0)

# Use adaptive thresholding with larger block size to reduce noise
# while still adapting to lighting conditions and alternating background
thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 21, 5)
cv2.imshow("Adaptive Threshold", thresh)
cv2.waitKey(0)

# invert so white boxes become black (easier to find as contours)
thresh_inv = cv2.bitwise_not(thresh)
cv2.imshow("Inverted Threshold", thresh_inv)
cv2.waitKey(0)

# IMPORTANT: Skip aggressive cleaning operations
# Instead, we will filter contours by their properties (size, shape, circularity)
# This preserves square details even on grey background areas
thresh_cleaned = thresh_inv.copy()
cv2.imshow("Final Threshold (No Cleaning)", thresh_cleaned)
cv2.waitKey(0)

# CROP TO RIGHT HALF - OMR checkboxes are on the right side of the page
# Get image dimensions
height, width = thresh_cleaned.shape
print(f"Image dimensions: {width} x {height}")

# crop to right half (full height, right half width)
mid_x = width // 2
thresh_cropped = thresh_cleaned[:, mid_x:]

print(f"Cropped to right half: from ({mid_x}, 0) to ({width}, {height})")
cv2.imshow("Right Half Cropped", thresh_cropped)
cv2.waitKey(0)

# find contours from cropped threshold
cnts = cv2.findContours(thresh_cropped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(f"Total contours found in right half: {len(cnts)}")

# filter contours to find ONLY the checkbox squares
# Checkboxes are small squares with specific size and aspect ratio
checkbox_contours = []
contour_info = []
rejected_info = []

for c in cnts:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    
    rejection_reason = None
    
    # skip if too small or too large
    if area < 40 or area > 6000:
        rejection_reason = f"area {area}"
    # checkboxes should be roughly square
    elif not (0.4 < (w / float(h) if h > 0 else 0) < 1.6):
        aspect_ratio = w / float(h) if h > 0 else 0
        rejection_reason = f"aspect_ratio {aspect_ratio:.2f}"
    else:
        # check solidity (filled area / bounding box area)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # RELAXED: checkboxes should be fairly solid (0.7)
        if solidity > 0.7:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                # adjust coordinates back to original image (add x offset for right half)
                contour_info.append({
                    'contour': c,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'x': x + mid_x,  # offset by crop position
                    'y': y,
                    'w': w,
                    'h': h
                })
        else:
            rejection_reason = f"solidity {solidity:.2f}"
    
    if rejection_reason:
        rejected_info.append((area, rejection_reason))

print(f"Found {len(contour_info)} potential checkbox squares")

# Debug: show what was rejected
if len(rejected_info) > 0 and len(rejected_info) <= 20:
    print(f"Sample rejected contours:")
    for area, reason in rejected_info[:10]:
        print(f"  - {reason}")

# filter by size consistency - all checkboxes should have similar area
if len(contour_info) > 0:
    areas = [info['area'] for info in contour_info]
    median_area = np.median(areas)
    std_area = np.std(areas)
    
    print(f"Area statistics: median={median_area:.0f}, std={std_area:.0f}, min={min(areas):.0f}, max={max(areas):.0f}")
    
    # RELAXED: keep contours with area within 2.0 std deviations (more lenient)
    for info in contour_info:
        if abs(info['area'] - median_area) < 2.0 * std_area:
            checkbox_contours.append(info['contour'])
        else:
            print(f"  Filtered out area {info['area']:.0f} (too far from median)")

print(f"After size filtering: {len(checkbox_contours)} checkbox squares")

if len(checkbox_contours) > 0:
    # visualize detected checkboxes in cropped area
    checkbox_image = cv2.cvtColor(thresh_cropped, cv2.COLOR_GRAY2BGR)
    
    for i, c in enumerate(checkbox_contours):
        x, y, w, h = cv2.boundingRect(c)
        color = (0, 255, 0)  # green for detected
        cv2.rectangle(checkbox_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(checkbox_image, str(i), (x+w//2-5, y+h//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    print(f"\nShowing {len(checkbox_contours)} detected checkboxes in right half area...")
    cv2.imshow("Detected Checkboxes (Right Half)", checkbox_image)
    cv2.waitKey(0)

# IMPORTANT: The layout is: multiple columns (questions) × 5 rows (options)
# Each column has 5 squares (one per option) stacked vertically
# We need to cluster x-positions to find distinct columns
checkbox_contours = sorted(checkbox_contours, key=lambda c: (cv2.boundingRect(c)[0], cv2.boundingRect(c)[1]))

if len(checkbox_contours) > 0:
    print(f"\nExpected: 33 questions × 5 options = 165 checkboxes")
    print(f"Detected: {len(checkbox_contours)} checkboxes")
    
    # Get bounding rectangles for all contours
    rects = [cv2.boundingRect(c) for c in checkbox_contours]
    
    # Cluster x-positions to find distinct columns (questions)
    # Use clustering with tolerance to handle slight misalignments
    x_coords = [rect[0] for rect in rects]
    x_coords_sorted = sorted(x_coords)
    
    column_clusters = []
    current_cluster = [x_coords_sorted[0]]
    COLUMN_TOLERANCE = 25  # tolerance for same column
    
    for x in x_coords_sorted[1:]:
        if x - current_cluster[-1] < COLUMN_TOLERANCE:
            current_cluster.append(x)
        else:
            column_clusters.append(current_cluster)
            current_cluster = [x]
    column_clusters.append(current_cluster)
    
    # Use median of each cluster as the column position
    x_positions = sorted([np.median(cluster) for cluster in column_clusters])
    
    # Find y-positions (rows/options) with clustering
    y_coords = [rect[1] for rect in rects]
    y_coords_sorted = sorted(y_coords)
    
    y_clusters = []
    current_cluster = [y_coords_sorted[0]]
    ROW_TOLERANCE = 20  # tolerance for same row
    
    for y in y_coords_sorted[1:]:
        if y - current_cluster[-1] < ROW_TOLERANCE:
            current_cluster.append(y)
        else:
            y_clusters.append(current_cluster)
            current_cluster = [y]
    y_clusters.append(current_cluster)
    
    y_positions = sorted([np.median(cluster) for cluster in y_clusters])
    
    print(f"Found {len(x_positions)} columns (questions) and {len(y_positions)} rows (options)")
    
    # organize into questions (iterate through x-positions for questions/columns)
    QUESTIONS = len(x_positions)
    OPTIONS = min(len(y_positions), 5)
    
    questions_data = []
    
    # Group squares by x-position (column/question) then by y-position (row/option)
    for q_idx in range(QUESTIONS):
        question_x = x_positions[q_idx]
        # tolerance for grouping squares in same column
        x_tolerance = 25
        
        # get all squares in this column (question)
        column_squares = []
        for c_idx, c in enumerate(checkbox_contours):
            x, y, w, h = rects[c_idx]
            if abs(x - question_x) < x_tolerance:
                column_squares.append((c, y))  # store contour and y position
        
        # sort by y-position (top to bottom) to get the options
        column_squares.sort(key=lambda item: item[1])
        
        marked_options = []
        for opt_idx, (checkbox, _) in enumerate(column_squares[:OPTIONS]):
            x, y, w, h = cv2.boundingRect(checkbox)
            
            # Use the inverted threshold image (where marked boxes are white/high value)
            # to check if the checkbox is marked
            checkbox_roi = thresh_cleaned[y:y+h, x:x+w]
            
            # count white pixels in the inverted threshold
            white_pixels = cv2.countNonZero(checkbox_roi)
            total_pixels = checkbox_roi.size
            filled_percentage = (white_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            # if more than 40% is white/filled, it's marked
            if filled_percentage > 40:
                marked_options.append(opt_idx + 1)
        
        questions_data.append({
            'question': q_idx + 1,
            'marked_options': marked_options
        })
    
    # display results
    print("\n" + "="*50)
    print("OMR SCANNING RESULTS")
    print("="*50)
    
    for q_data in questions_data:
        marked = q_data['marked_options']
        if marked:
            print(f"Q{q_data['question']:2d}: Option(s) {marked} marked")
        else:
            print(f"Q{q_data['question']:2d}: No option marked")
    
    # visualize detected checkboxes
    checkbox_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for i, c in enumerate(checkbox_contours):
        x, y, w, h = cv2.boundingRect(c)
        
        # check if marked
        checkbox_roi = gray[y:y+h, x:x+w]
        avg_intensity = np.mean(checkbox_roi)
        is_marked = avg_intensity < 150
        
        # draw rectangle with color coding
        color = (0, 0, 255) if is_marked else (0, 255, 0)  # red=marked, green=empty
        cv2.rectangle(checkbox_image, (x, y), (x+w, y+h), color, 2)
        
        # add label
        question_num = (i // OPTIONS) + 1
        option_num = (i % OPTIONS) + 1
        cv2.putText(checkbox_image, f"Q{question_num}O{option_num}", (x-20, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    print(f"\nDisplaying {len(checkbox_contours)} detected checkboxes...")
    print("(Green = Empty, Red = Marked)")
    cv2.imshow("Detected Checkboxes", checkbox_image)
    cv2.waitKey(0)
else:
    print("ERROR: No checkbox squares detected!")
    print("Displaying original image for inspection...")
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
