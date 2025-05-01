import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
import string

# Step 1: Load the image
image = cv2.imread('img.png')

# Step 2: Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Thresholding to get a binary image (Black and White)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

# Step 4: Resize and Normalize Image
resized_image = cv2.resize(binary_image, (256, 256))  # Adjust size as needed
normalized_image = resized_image / 255.0  # Normalize the pixel values to 0-1

# Step 5: Noise Removal (Optional, but can help improve accuracy)
denoised_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)

# Step 6: Segmenting the image (Optional: Break into individual characters/words)
# Find contours in the binary image to segment the words or characters
contours, _ = cv2.findContours(denoised_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for displaying contours
segmented_image = image.copy()

# Draw contours (this will give you the bounding boxes for segmentation)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(segmented_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the segmented image with contours
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("Segmented Image with Contours")
plt.axis('off')
# plt.show()

# Optional: Extract the segmented regions for further processing
segmented_regions = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    segment = binary_image[y:y+h, x:x+w]
    segmented_regions.append(segment)

# Display segmented regions (characters or words)
for idx, region in enumerate(segmented_regions):
    plt.subplot(1, len(segmented_regions), idx+1)
    plt.imshow(region, cmap='gray')
    plt.title(f"Region {idx+1}")
    plt.axis('off')
# plt.show()


def extract_hog_features(segments):
    hog_features = []
    for segment in segments:
        # Resize to a standard size (e.g., 64x64) to keep dimensions consistent
        resized_segment = cv2.resize(segment, (64, 64))

        # Calculate HOG features (remove multichannel argument)
        fd, hog_image = hog(resized_segment, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)

        # Enhance the visualization of HOG features (optional)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        # Store the HOG feature descriptor
        hog_features.append(fd)

        # Display the HOG visualization (optional)
        plt.imshow(hog_image_rescaled, cmap='gray')
        plt.title("HOG Features")
        plt.axis('off')
        # plt.show()

    return np.array(hog_features)

# Extract HOG features from the segmented regions
hog_features = extract_hog_features(segmented_regions)

# Check the shape of the feature vector
print(f"HOG Feature vector shape: {hog_features.shape}")


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import string

# Assuming you have already obtained the segmented regions and their corresponding HOG features
# Example segmented regions and HOG features after segmentation and HOG extraction
segmented_regions = [
    ['H', 'E', 'L', 'L', 'O'],
    ['A', 'B', 'C', 'D', 'E'],
    ['1', '2', '3', '4', '5']
]  # Replace with actual segmented regions
hog_features = [
    np.random.rand(36),  # Placeholder for actual HOG features of segment 1
    np.random.rand(36),  # Placeholder for actual HOG features of segment 2
    np.random.rand(36)   # Placeholder for actual HOG features of segment 3
]  # Replace with actual HOG features

# Define the possible labels (letters, digits, punctuation)
possible_labels = list(string.ascii_letters + string.digits + " .,!?:;\"'()[]{}<>@#%^&*")

# Convert the segmented regions into a sequence of labels (handling missing characters)
def handle_missing_labels(sequence, max_length=20):
    # Placeholder for missing characters
    labels = sequence + ['<MISSING>'] * (max_length - len(sequence))
    return labels[:max_length]  # Truncate if longer than max length

# Example with missing characters in segmented regions (let's assume missing characters in the second and third segments)
labels_1 = handle_missing_labels(['H', 'E', 'L', 'L', 'O'])
labels_2 = handle_missing_labels(['A', 'B', 'C', 'D', 'E'])
labels_3 = handle_missing_labels(['1', '2', '3'])  # Only 3 characters, padding with <MISSING>

# Combine all the label sequences
all_labels = labels_1 + labels_2 + labels_3
all_hog_features = hog_features  # Add actual HOG features here

# Create a binary mask for each segment
def create_mask(sequence):
    return [1 if char != '<MISSING>' else 0 for char in sequence]

# Example binary mask for each sequence
mask_1 = create_mask(labels_1)
mask_2 = create_mask(labels_2)
mask_3 = create_mask(labels_3)

# Combine all masks
all_masks = mask_1 + mask_2 + mask_3

# Step 1: Split the dataset into training and testing sets (using a simple example here)
X_train, X_test, y_train, y_test = train_test_split(all_hog_features, all_labels, test_size=0.3, random_state=42)

# Step 2: Train an SVM classifier
svm_classifier = SVC(kernel='linear')  # You can try different kernels (linear, rbf, etc.)
svm_classifier.fit(X_train, y_train)

# Step 3: Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Step 4: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Predicting new segments
new_sample = np.array([np.random.rand(36)])  # Example new segment HOG features
prediction = svm_classifier.predict(new_sample)
print(f"Predicted Label: {prediction[0]}")



# Function to extract HOG features dynamically from segmented regions
def extract_hog_features_from_segment(segment_image):
    # Example of HOG extraction (resize and calculate HOG)
    resized_segment = cv2.resize(segment_image, (64, 128))  # Resize if necessary
    fd, hog_image = hog(resized_segment, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd  # Return HOG feature vector


# Dynamically process images and their segments
def process_image_and_extract_features(image_segments, labels):
    all_hog_features = []
    all_labels = []

    for segment, label in zip(image_segments, labels):
        hog_features = extract_hog_features_from_segment(segment)
        all_hog_features.append(hog_features)
        all_labels.append(label)

    return np.array(all_hog_features), all_labels


# Example function to handle dynamic segmentation
def segment_image(image):
    # This function should dynamically segment your image and return each individual character as a segment
    # For simplicity, let's assume each image is already pre-segmented into character regions (a list of images)
    segmented_regions = [image]  # Replace with actual segmentation logic
    return segmented_regions


# Dynamically handle random image and corresponding labels
image = cv2.imread('random_image.png')  # Read your input image (use actual image path)
segmented_regions = segment_image(image)  # Dynamically segment the image into character regions

# Random example of labels for the segmented regions (this should come from your image processing or OCR)
labels = ['H', 'E', 'L', 'L', 'O']  # For example, adjust dynamically based on segmentation

# Process image and extract HOG features dynamically for all segments
all_hog_features, all_labels = process_image_and_extract_features(segmented_regions, labels)

# Ensure that the number of labels matches the number of HOG features
assert len(all_hog_features) == len(
    all_labels), f"Mismatch between feature vectors ({len(all_hog_features)}) and labels ({len(all_labels)})"

# Step 1: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_hog_features, all_labels, test_size=0.3, random_state=42)

# Step 2: Train an SVM classifier
svm_classifier = SVC(kernel='linear')  # You can try different kernels (linear, rbf, etc.)
svm_classifier.fit(X_train, y_train)

# Step 3: Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Step 4: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Predicting new segments
new_sample = np.array([np.random.rand(36)])  # Example new segment HOG features
prediction = svm_classifier.predict(new_sample)
print(f"Predicted Label: {prediction[0]}")
