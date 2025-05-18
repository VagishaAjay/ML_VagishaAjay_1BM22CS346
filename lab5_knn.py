import math

# Dataset: each point is (x1, x2, label)
dataset = [
    (1, 2, 'A'),
    (2, 3, 'A'),
    (3, 6, 'B'),
    (6, 8, 'B'),
    (7, 7, 'B')
]

# Test point
test_point = (4, 5)

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Compute distances
distances = []
for data in dataset:
    dist = euclidean_distance(test_point, (data[0], data[1]))
    distances.append((dist, data))
    print(f"Distance from {test_point} to {data[:2]} (label={data[2]}): {dist:.4f}")

# Sort by distance
distances.sort(key=lambda x: x[0])

print("\n--- 3 Nearest Neighbors ---")
for i in range(3):
    dist, data = distances[i]
    print(f"{i+1}) Point: {data[:2]}, Label: {data[2]}, Distance: {dist:.4f}")
