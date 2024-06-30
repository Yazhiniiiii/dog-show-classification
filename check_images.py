import os
import time
from classifier import classify_image

# Directory containing images for testing
image_dir = 'images/'

# List of image files
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Models to evaluate
models = ['alexnet', 'vgg', 'resnet']

# Initialize results dictionary
results = {
    'alexnet': {'not_a_dog_correct': 0, 'dog_correct': 0, 'breed_correct': 0, 'total_dog_images': 0, 'total_images': 0, 'time_taken': 0},
    'vgg': {'not_a_dog_correct': 0, 'dog_correct': 0, 'breed_correct': 0, 'total_dog_images': 0, 'total_images': 0, 'time_taken': 0},
    'resnet': {'not_a_dog_correct': 0, 'dog_correct': 0, 'breed_correct': 0, 'total_dog_images': 0, 'total_images': 0, 'time_taken': 0}
}

# Function to evaluate a model
def evaluate_model(model_name):
    start_time = time.time()
    total_dog_images = 0
    total_images = len(image_files)

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label, confidence = classify_image(image_path, model_name)
        
        if 'dog' in image_file:
            total_dog_images += 1
            if 'Dog' in label:
                results[model_name]['dog_correct'] += 1
                # Check breed accuracy (simplified)
                if label.split()[0].lower() in image_file.lower():
                    results[model_name]['breed_correct'] += 1
        else:
            if 'Not a Dog' in label:
                results[model_name]['not_a_dog_correct'] += 1

    results[model_name]['total_dog_images'] = total_dog_images
    results[model_name]['total_images'] = total_images
    results[model_name]['time_taken'] = time.time() - start_time

# Evaluate each model
for model in models:
    evaluate_model(model)

# Print results in a table format
print("Results Table")
print(f"{'CNN Model Architecture:':<25} {'% Not-a-Dog Correct':<20} {'% Dogs Correct':<15} {'% Breeds Correct':<18} {'% Match Labels':<15}")
for model in models:
    not_a_dog_accuracy = (results[model]['not_a_dog_correct'] / (results[model]['total_images'] - results[model]['total_dog_images'])) * 100
    dog_accuracy = (results[model]['dog_correct'] / results[model]['total_images']) * 100
    breed_accuracy = (results[model]['breed_correct'] / results[model]['total_dog_images']) * 100 if results[model]['total_dog_images'] > 0 else 0
    match_labels = (results[model]['breed_correct'] / results[model]['total_images']) * 100
    
    print(f"{model.capitalize():<25} {not_a_dog_accuracy:<20.1f} {dog_accuracy:<15.1f} {breed_accuracy:<18.1f} {match_labels:<15.1f}")
