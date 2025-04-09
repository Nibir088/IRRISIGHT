import numpy as np
from utils.metrics import *

np.random.seed(42)
# Step 1: Generate sample data
num_classes = 3  # Example number of classes
batch_size = 4   # Batch size
height, width = 4, 4  # Image dimensions

# Ground truth (y_true) with shape (batch_size, height, width)
y_true = np.random.randint(0, num_classes, size=(8, height, width))

# Predicted probabilities (y_pred_probs) with shape (batch_size, num_classes, height, width)
y_pred_probs = np.random.rand(8, num_classes, height, width)
y_pred_probs /= np.sum(y_pred_probs, axis=1, keepdims=True)  # Normalize to make probabilities

# Split into two batches
y_true_batches = np.split(y_true, batch_size)
y_pred_batches = np.split(y_pred_probs, batch_size)
# print(y_true_batches)
# Step 2: Initialize SegmentationMetrics
metrics = SegmentationMetrics(num_classes)

y_true_batches[0][y_true_batches[0]==2] = 0

# Step 3: Update metrics batch by batch and collect confusion matrix
for batch_idx, (y_true_batch, y_pred_batch) in enumerate(zip(y_true_batches, y_pred_batches)):
    metrics.update(y_true_batch, y_pred_batch)
    # Access confusion matrix after processing the batch
    batch_conf_matrix = metrics.metrics['precision']['micro'].conf_matrix.compute()
    print(f"\nConfusion Matrix after Batch {batch_idx + 1}:")
    print(batch_conf_matrix)

# Step 4: Access and print the final confusion matrix
final_conf_matrix = metrics.metrics['precision']['micro'].conf_matrix.compute()
print("\nFinal Confusion Matrix:")
print(final_conf_matrix)

# Step 5: Compute and display all metric results
results = metrics.compute()

for metric_name, values in results.items():
    print(f"\nMetric: {metric_name}")
    for avg_type, value in values.items():
        print(f"  {avg_type}: {value}")
