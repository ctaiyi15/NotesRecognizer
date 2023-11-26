import numpy as np
from skimage import io, measure, filters, morphology
from skimage.measure import label
from collections import OrderedDict
import matplotlib.pyplot as plt

def find_shapes(binary):
    # # Load the image 
    # image = io.imread(image_path, as_gray=True)

    # # Apply a threshold to convert the image to binary
    # binary = image < filters.threshold_otsu(image)

    # remove small noise
    # binary = morphology.opening(binary, morphology.disk(2))

    # ensure parts of the same digit are not separated
    binary = morphology.dilation(binary, morphology.disk(3))

    io.imshow(binary, cmap='gray')

    # Label 
    labels = label(binary)

    # Create a list to store the centroids
    centroids = []
    widths = []
    # Iterate over the properties of each labeled region
    for region in measure.regionprops(labels):
        # Skip small regions
        if region.area >= 200:
            # Append the centroid and width of the region to the list
            centroids.append((region.centroid, region.bbox))
            minr, minc, maxr, maxc = region.bbox
            widths.append(maxc - minc)

    # Compute the average width
    avg_width = np.mean(widths)

    # Group centroids by their vertical and horizontal position with a tolerance
    rows = OrderedDict()
    tolerance = 200  # adjust the tolerance to control the row height
    for centroid, bbox in centroids:
        placed = False
        for key in rows:
            if abs(centroid[0] - key) < tolerance:
                rows[key].append((centroid, bbox))
                placed = True
                break
        if not placed:
            rows[centroid[0]] = [(centroid, bbox)]

    # Sort the rows by the key (vertical position)
    rows = OrderedDict(sorted(rows.items()))

    # Sort each group of centroids by their horizontal position and then by their vertical position
    for key in rows:
        rows[key].sort(key=lambda x: (x[0][1], x[0][0]))

    # Flatten the dictionary into a list
    sorted_centroids = [item for sublist in rows.values() for item in sublist]

    # Check and add missing centroids horizontally and vertically
    final_centroids = []
    prev_centroid = None
    prev_row_key = None
    for centroid, bbox in sorted_centroids:
        row_key = next((key for key in rows if (centroid, bbox) in rows[key]), None)
        if prev_row_key is not None:
            # Check vertical distance
            if row_key is not None and row_key - prev_row_key > 200:  # 200 can be change by picture change
                final_centroids.append(((-2, -2), None))  
                prev_row_key = row_key
                # Fall through to the next check to handle the first centroid of the new row

        if prev_centroid is not None:
            # Check horizontal distance
            horizontal_distance = centroid[1] - prev_centroid[1]  # Use centroid[1] for x coordinate
            num_inserts = int(np.floor(horizontal_distance / avg_width)) - 1  # Replace avg_width with your threshold for data change
            if num_inserts > 0:
                final_centroids.extend([((-1, -1), None)] * num_inserts)
                if centroid == (-2, -2):
                    prev_centroid = None  # Reset prev_centroid after handling the first centroid of the new row

        final_centroids.append((centroid, bbox))
        prev_row_key = row_key
        prev_centroid = centroid
 
    # Return the final list of centroids, rounding each to the nearest integer
    return [((round(centroid[0]), round(centroid[1])), bbox) if centroid not in [(-2, -2), (-1, -1)] else (centroid, bbox) for centroid, bbox in final_centroids]

def normalize_moments(moments):
    norm = np.linalg.norm(moments)
    return moments / norm if norm != 0 else moments

def cosine_similarity(moments1, moments2):
    hu_moments1_normalized = normalize_moments(moments1)
    hu_moments2_normalized = normalize_moments(moments2)
    
    similarity = np.dot(hu_moments1_normalized, hu_moments2_normalized)
    return similarity

def show_centroid(img, centroids):
    plt.imshow(img,cmap='gray')
    for index, (centroid, _) in enumerate(centroids):
        if centroid not in [(-1, -1), (-2, -2)]:
            plt.plot(centroid[1], centroid[0], 'ro', markersize=2)
            plt.text(centroid[1] + 5, centroid[0] + 5, str(index), color='yellow', fontsize=8)
    plt.show()
