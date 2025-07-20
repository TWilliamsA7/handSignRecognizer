import os
import random

def balance_labels(maxCount: int, processed_dir: str = "data/processed"):
    processedDirs = dict()
    sampMin = 1000000

    # Create a list of the label directories
    for label in os.listdir(processed_dir):
        path = os.path.join(processed_dir, label)
        numSamples = len(os.listdir(path))
        processedDirs[path] = numSamples
        if (numSamples < sampMin): 
            sampMin = numSamples
        
    for path, numSamples in processedDirs.items():
        sampleNames = set(os.listdir(path))
        mustDelete = numSamples - sampMin
        for _ in range(mustDelete):
            pathToDelete = sampleNames.pop()
            print(f"Removing {path}/{pathToDelete}!")
            os.remove(f"{path}/{pathToDelete}")
        
    print("Sample Classes Have Been Successfully Balanced!")