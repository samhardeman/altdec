import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_crowd_level(empty_image_path, current_image_path, roi_mask=None, threshold_sensitivity=25):
    """
    Improved crowd detection using advanced background subtraction and contour filtering.
    
    Args:
        empty_image_path: Path to the image without people (baseline)
        current_image_path: Path to the current image to analyze
        roi_mask: Optional binary mask for region of interest (255 for areas to analyze)
        threshold_sensitivity: Lower values make the detection more sensitive
        
    Returns:
        float: Percentage of the image covered by people (0-100%)
        str: Busy level description
        np.ndarray: Mask of detected areas
    """
    # Load images
    empty_img = cv2.imread(empty_image_path)
    current_img = cv2.imread(current_image_path)
    
    # Ensure images are the same size
    if empty_img.shape != current_img.shape:
        current_img = cv2.resize(current_img, (empty_img.shape[1], empty_img.shape[0]))
    
    # Create default ROI mask if none provided (whole image)
    if roi_mask is None:
        roi_mask = np.ones(empty_img.shape[:2], dtype=np.uint8) * 255

    # Convert to HSV color space and split channels
    empty_hsv = cv2.cvtColor(empty_img, cv2.COLOR_BGR2HSV)
    current_hsv = cv2.cvtColor(current_img, cv2.COLOR_BGR2HSV)
    
    # Apply stronger Gaussian blur to reduce noise and lighting variations
    empty_blur = cv2.GaussianBlur(empty_hsv, (7, 7), 0)
    current_blur = cv2.GaussianBlur(current_hsv, (7, 7), 0)
    
    # Split channels
    h1, s1, v1 = cv2.split(empty_blur)
    h2, s2, v2 = cv2.split(current_blur)
    
    # Calculate weighted differences with adjusted weights
    diff_h = cv2.absdiff(h1, h2) * 0.3  # Reduce hue importance since it varies with lighting
    diff_s = cv2.absdiff(s1, s2) * 1.2  # Increase saturation weight
    diff_v = cv2.absdiff(v1, v2) * 0.8  # Reduce value weight to minimize lighting impact
    
    # Normalize value channels to reduce lighting impact
    v1_norm = cv2.normalize(v1, None, 0, 255, cv2.NORM_MINMAX)
    v2_norm = cv2.normalize(v2, None, 0, 255, cv2.NORM_MINMAX)
    diff_v_norm = cv2.absdiff(v1_norm, v2_norm) * 0.8

    # Combine differences with adjusted weights
    diff_combined = cv2.addWeighted(diff_s, 0.6, diff_v_norm, 0.4, 0)
    diff_combined = cv2.addWeighted(diff_combined, 0.8, diff_h, 0.2, 0)

    # Apply histogram equalization to handle lighting variations
    diff_combined = cv2.equalizeHist(diff_combined.astype(np.uint8))

    # Convert to 8-bit unsigned integer
    diff_combined = diff_combined.astype(np.uint8)

    # Apply adaptive thresholding instead of fixed threshold
    thresholded = cv2.adaptiveThreshold(
        diff_combined,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # Constant subtracted from mean
    )

    # Improve morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # Reduced from 15,15
    
    # Apply morphological operations
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Apply ROI mask
    if roi_mask is not None:
        closed = cv2.bitwise_and(closed, closed, mask=roi_mask)

    # Rest of the existing code remains the same...
    
    # Find contours to filter by size
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for filtered contours
    filtered_mask = np.zeros_like(closed)
    
    # Filter contours by size (remove very small ones that are likely noise)
    min_contour_area = 100  # Minimum area to be considered a significant difference
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    
    # Calculate coverage only in the ROI area
    roi_pixel_count = np.sum(roi_mask == 255)
    white_pixel_count = np.sum(filtered_mask == 255)
    
    if roi_pixel_count > 0:
        percentage = (white_pixel_count / roi_pixel_count) * 100
    else:
        percentage = 0
    
    # Determine busy level
    if percentage < 10:
        busy_level = "Not busy"
    elif percentage < 25:
        busy_level = "Slightly busy"
    elif percentage < 40:
        busy_level = "Moderately busy"
    elif percentage < 60:
        busy_level = "Very busy"
    else:
        busy_level = "Extremely busy"
    
    return percentage, busy_level, filtered_mask

def create_roi_mask(image_shape, exclude_top=0.0, exclude_bottom=0.0, exclude_left=0.0, exclude_right=0.0):
    """
    Creates a region of interest mask, excluding specified percentages from edges.
    Useful for excluding areas like counters, displays, or ceilings.
    
    Args:
        image_shape: Shape of the image (height, width)
        exclude_top, exclude_bottom, exclude_left, exclude_right: Percentage of image to exclude (0.0-1.0)
        
    Returns:
        np.ndarray: Binary mask with ROI areas set to 255
    """
    height, width = image_shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Calculate excluded regions
    top_pixels = int(height * exclude_top)
    bottom_pixels = int(height * exclude_bottom)
    left_pixels = int(width * exclude_left)
    right_pixels = int(width * exclude_right)
    
    # Set excluded regions to 0
    if top_pixels > 0:
        mask[:top_pixels, :] = 0
    if bottom_pixels > 0:
        mask[-bottom_pixels:, :] = 0
    if left_pixels > 0:
        mask[:, :left_pixels] = 0
    if right_pixels > 0:
        mask[:, -right_pixels:] = 0
        
    return mask

def visualize_improved_results(empty_image_path, current_image_path, roi_exclude=None, threshold_sensitivity=25):
    """
    Visualizes the improved crowd detection process with images.
    
    Args:
        empty_image_path: Path to the image without people
        current_image_path: Path to the current image to analyze
        roi_exclude: Dict with exclude percentages for top, bottom, left, right
        threshold_sensitivity: Sensitivity for threshold detection
        
    Returns:
        float: Crowd percentage
    """
    # Load images
    empty_img = cv2.imread(empty_image_path)
    current_img = cv2.imread(current_image_path)
    
    # Ensure images are the same size
    if empty_img.shape != current_img.shape:
        current_img = cv2.resize(current_img, (empty_img.shape[1], empty_img.shape[0]))
    
    # Create ROI mask
    if roi_exclude is None:
        roi_exclude = {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}
    
    roi_mask = create_roi_mask(
        empty_img.shape, 
        exclude_top=roi_exclude.get('top', 0.0),
        exclude_bottom=roi_exclude.get('bottom', 0.0),
        exclude_left=roi_exclude.get('left', 0.0),
        exclude_right=roi_exclude.get('right', 0.0)
    )
    
    # Run detection
    percentage, busy_level, filtered_mask = detect_crowd_level(
        empty_image_path, 
        current_image_path, 
        roi_mask=roi_mask,
        threshold_sensitivity=threshold_sensitivity
    )
    
    # Create visual mask for ROI
    roi_visual = np.zeros_like(empty_img)
    roi_visual[roi_mask == 255] = [0, 255, 0]  # Green for ROI areas
    roi_overlay = cv2.addWeighted(empty_img, 0.7, roi_visual, 0.3, 0)
    
    # Create visual mask for detections
    detection_visual = np.zeros_like(current_img)
    detection_visual[filtered_mask == 255] = [0, 0, 255]  # Red for detected areas
    detection_overlay = cv2.addWeighted(current_img, 0.7, detection_visual, 0.3, 0)
    
    # Convert images for matplotlib (BGR to RGB)
    empty_img_rgb = cv2.cvtColor(empty_img, cv2.COLOR_BGR2RGB)
    current_img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
    roi_overlay_rgb = cv2.cvtColor(roi_overlay, cv2.COLOR_BGR2RGB)
    detection_overlay_rgb = cv2.cvtColor(detection_overlay, cv2.COLOR_BGR2RGB)
    
    # Find contours and draw them for visualization
    contours_image = current_img.copy()
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contours_image, contours, -1, (0, 255, 255), 2)
    contours_image_rgb = cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB)
    
    # Create visualization with 6 panels
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    plt.title("Empty Image (Baseline)")
    plt.imshow(empty_img_rgb)
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    plt.title("Current Image")
    plt.imshow(current_img_rgb)
    plt.axis('off')
    
    plt.subplot(3, 2, 3)
    plt.title("Region of Interest")
    plt.imshow(roi_overlay_rgb)
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.title("Detected Differences")
    plt.imshow(filtered_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 2, 5)
    plt.title("Contour Detection")
    plt.imshow(contours_image_rgb)
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    plt.title(f"Final Overlay (Crowd: {percentage:.1f}%, {busy_level})")
    plt.imshow(detection_overlay_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("improved_crowd_detection.png")
    plt.show()
    
    return percentage

def main():
    # Replace these with your actual image paths
    empty_image_path = "habit-baseline.jpg"  # Image with no people
    current_image_path = "habit-variance.png"  # Current image to analyze
    
    # Define regions to exclude from analysis (e.g., counter areas, displays)
    # Adjust these values based on your specific scene
    roi_exclude = {
        'top': 0.05,     # Exclude top 5% of image
        'bottom': 0.05,  # Exclude bottom 5% of image
        'left': 0.05,    # Exclude left 5% of image
        'right': 0.05    # Exclude right 5% of image
    }
    
    # Adjust sensitivity (lower = more sensitive)
    threshold_sensitivity = 25
    
    # Run the improved detection and visualization
    percentage = visualize_improved_results(
        empty_image_path, 
        current_image_path,
        roi_exclude=roi_exclude,
        threshold_sensitivity=threshold_sensitivity
    )
    
    print(f"Crowd coverage: {percentage:.1f}%")
    
    # For real-time application, you could also run just the detection function
    percentage, busy_level, _ = detect_crowd_level(
        empty_image_path, 
        current_image_path,
        threshold_sensitivity=threshold_sensitivity
    )
    print(f"Busy level: {busy_level}")

if __name__ == "__main__":
    main()