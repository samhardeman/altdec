import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_crowd_level(empty_image_path, current_image_path, threshold=30, blur_size=5):
    """
    Detects the crowd level by comparing an empty image to the current image.
    
    Args:
        empty_image_path: Path to the image without people (baseline)
        current_image_path: Path to the current image to analyze
        threshold: Pixel difference threshold to consider as a change
        blur_size: Size of Gaussian blur to apply
        
    Returns:
        float: Percentage of the image covered by people (0-100%)
        str: Busy level description
    """
    # Load images
    empty_img = cv2.imread(empty_image_path)
    current_img = cv2.imread(current_image_path)
    
    # Ensure images are the same size
    if empty_img.shape != current_img.shape:
        current_img = cv2.resize(current_img, (empty_img.shape[1], empty_img.shape[0]))
    
    # Convert to grayscale
    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    empty_blur = cv2.GaussianBlur(empty_gray, (blur_size, blur_size), 0)
    current_blur = cv2.GaussianBlur(current_gray, (blur_size, blur_size), 0)
    
    # Calculate absolute difference between the two images
    diff = cv2.absdiff(empty_blur, current_blur)
    
    # Apply threshold to get binary image
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # Calculate percentage of white pixels (changed areas)
    white_pixel_count = np.sum(thresholded == 255)
    total_pixels = thresholded.shape[0] * thresholded.shape[1]
    percentage = (white_pixel_count / total_pixels) * 100
    
    # Determine busy level
    if percentage < 10:
        busy_level = "Not busy"
    elif percentage < 30:
        busy_level = "Slightly busy"
    elif percentage < 50:
        busy_level = "Moderately busy"
    elif percentage < 70:
        busy_level = "Very busy"
    else:
        busy_level = "Extremely busy"
    
    return percentage, busy_level

def visualize_results(empty_image_path, current_image_path, threshold=30, blur_size=5):
    """
    Visualizes the crowd detection process with images.
    """
    # Load images
    empty_img = cv2.imread(empty_image_path)
    current_img = cv2.imread(current_image_path)
    
    # Ensure images are the same size
    if empty_img.shape != current_img.shape:
        current_img = cv2.resize(current_img, (empty_img.shape[1], empty_img.shape[0]))
    
    # Convert to grayscale
    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    empty_blur = cv2.GaussianBlur(empty_gray, (blur_size, blur_size), 0)
    current_blur = cv2.GaussianBlur(current_gray, (blur_size, blur_size), 0)
    
    # Calculate difference
    diff = cv2.absdiff(empty_blur, current_blur)
    
    # Apply threshold
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    # Calculate crowd percentage
    white_pixel_count = np.sum(processed == 255)
    total_pixels = processed.shape[0] * processed.shape[1]
    percentage = (white_pixel_count / total_pixels) * 100
    
    # Create visual mask
    mask_colored = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    mask_colored[processed == 255] = [0, 0, 255]  # Red for detected areas
    
    # Overlay on current image
    overlay = cv2.addWeighted(current_img, 0.7, mask_colored, 0.3, 0)
    
    # Convert images for matplotlib (BGR to RGB)
    empty_img_rgb = cv2.cvtColor(empty_img, cv2.COLOR_BGR2RGB)
    current_img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Empty Image (Baseline)")
    plt.imshow(empty_img_rgb)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Current Image")
    plt.imshow(current_img_rgb)
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("Difference Detection")
    plt.imshow(processed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title(f"Overlay (Crowd: {percentage:.1f}%)")
    plt.imshow(overlay_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("crowd_detection_visualization.png")
    plt.show()
    
    return percentage

def main():
    # Replace these with your actual image paths
    empty_image_path = "habit-baseline.jpg"  # Image with no people
    current_image_path = "habit-variance.png"  # Current image to analyze
    
    # Get crowd percentage and level
    percentage, busy_level = detect_crowd_level(empty_image_path, current_image_path)
    
    print(f"Crowd coverage: {percentage:.1f}%")
    print(f"Busy level: {busy_level}")
    
    # Visualize results
    visualize_results(empty_image_path, current_image_path)

if __name__ == "__main__":
    main()