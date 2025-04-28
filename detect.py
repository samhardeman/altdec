import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_crowd_level(empty_image_path, current_image_path, threshold=30, blur_size=5):
    empty_img = cv2.imread(empty_image_path)
    current_img = cv2.imread(current_image_path)

    if empty_img.shape != current_img.shape:
        current_img = cv2.resize(current_img, (empty_img.shape[1], empty_img.shape[0]))

    # Create mask from green area in the empty image
    hsv = cv2.cvtColor(empty_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Grayscale and blur
    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    empty_blur = cv2.GaussianBlur(empty_gray, (blur_size, blur_size), 0)
    current_blur = cv2.GaussianBlur(current_gray, (blur_size, blur_size), 0)

    # Difference detection
    diff = cv2.absdiff(empty_blur, current_blur)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # Only consider changes within the green mask
    masked_diff = cv2.bitwise_and(thresholded, thresholded, mask=green_mask)

    white_pixel_count = np.sum(masked_diff == 255)
    total_pixels = np.sum(green_mask > 0)
    percentage = (white_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

    if percentage < 20:
        busy_level = "Not busy"
    elif percentage < 40:
        busy_level = "Slightly busy"
    elif percentage < 60:
        busy_level = "Moderately busy"
    elif percentage < 80:
        busy_level = "Very busy"
    else:
        busy_level = "Extremely busy"

    return percentage, busy_level


def visualize_results(empty_image_path, current_image_path, percentage, busy_level):
    empty_img = cv2.imread(empty_image_path)
    current_img = cv2.imread(current_image_path)

    if empty_img.shape != current_img.shape:
        current_img = cv2.resize(current_img, (empty_img.shape[1], empty_img.shape[0]))

    # Make the green mask from the empty image
    hsv = cv2.cvtColor(empty_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Difference detection (same as before)
    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(empty_gray, current_gray)
    _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # Apply green mask to restrict changes
    masked_diff = cv2.bitwise_and(thresholded, thresholded, mask=green_mask)

    # Create red overlay for detected areas
    overlay_img = current_img.copy()
    red_overlay = np.zeros_like(current_img, dtype=np.uint8)
    red_overlay[:, :] = (0, 0, 255)  # Red in BGR
    red_overlay_masked = cv2.bitwise_and(red_overlay, red_overlay, mask=masked_diff)
    overlay_img = cv2.addWeighted(overlay_img, 1.0, red_overlay_masked, 0.5, 0)

    # Combine results in one display
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].imshow(cv2.cvtColor(empty_img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Empty Image (Baseline)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Current Image")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(masked_diff, cmap="gray")
    axs[1, 0].set_title("Masked Difference Detection")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title(f"Overlay (Crowd: {percentage:.1f}%, {busy_level})")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    # Replace these with your actual image paths
    empty_image_path = "./bounded/pitajungle.jpg"  # Image with no people
    current_image_path = "./early/pitajungle.jpeg"  # Current image to analyze
    
    # Get crowd percentage and level
    percentage, busy_level = detect_crowd_level(empty_image_path, current_image_path)
    
    print(f"Crowd coverage: {percentage:.1f}%")
    print(f"Busy level: {busy_level}")
    
    # Visualize results
    visualize_results(empty_image_path, current_image_path, percentage, busy_level)

if __name__ == "__main__":
    main()