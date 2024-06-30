import os
import time
from classifier import load_model, classify_image, transform

image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]  # Add your image paths here
labels = ["dog", "not-a-dog", ...]  # Corresponding labels

def evaluate_model(architecture):
    model = load_model(architecture)
    correct_dog = 0
    correct_not_dog = 0
    correct_breed = 0
    total_dogs = 0
    total_not_dogs = 0
    
    start_time = time.time()
    for i, image_path in enumerate(image_paths):
        label = labels[i]
        predicted = classify_image(image_path, model, transform)
        
        if label == "dog":
            total_dogs += 1
            if predicted == 1:  # Assuming '1' represents dog in the classifier's output
                correct_dog += 1
                if predicted_breed == expected_breed:  # Check breed accuracy
                    correct_breed += 1
        else:
            total_not_dogs += 1
            if predicted == 0:  # Assuming '0' represents not-a-dog
                correct_not_dog += 1
    
    total_time = time.time() - start_time
    
    return {
        "correct_dog": correct_dog / total_dogs * 100,
        "correct_not_dog": correct_not_dog / total_not_dogs * 100,
        "correct_breed": correct_breed / total_dogs * 100,
        "total_time": total_time,
    }

results = {}
for architecture in ["resnet", "alexnet", "vgg"]:
    results[architecture] = evaluate_model(architecture)

print("Results Table:")
print(results)

