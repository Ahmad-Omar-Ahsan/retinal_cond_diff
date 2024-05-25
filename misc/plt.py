import matplotlib.pyplot as plt
import numpy as np

# Data from the dictionaries
correct_classification = {
    'AMD': 16,
    'Cataract': 18,
    'DR': 298,
    'Glaucoma': 132,
    'Myopia': 32,
    'Normal': 137
}

total_labels_per_class = {
    'AMD': 32,
    'Cataract': 20,
    'DR': 347,
    'Glaucoma': 164,
    'Myopia': 37,
    'Normal': 167
}

# Extract keys and values
labels = list(correct_classification.keys())
correct_values = list(correct_classification.values())
total_values = list(total_labels_per_class.values())

# Set up the figure
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, correct_values, width, label='Correct Classification', color='b', alpha=0.7)
plt.bar(x + width/2, total_values, width, label='Total Labels per Class', color='g', alpha=0.7)

# Add labels and title
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Correct Classification vs. Total Labels per Class')
plt.xticks(x, labels)
plt.legend()

# Show the plot
plt.show()
plt.savefig('Correct vs Total.png')