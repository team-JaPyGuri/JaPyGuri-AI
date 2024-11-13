# make file path csv
import pandas as pd
import os

def make_file_path(root):
    image_dir = root + "/images"
    label_dir = root + "/labels"
    
    image_paths = []
    label_paths = []

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'): 
                image_paths.append(os.path.join(image_dir, file))
                label_paths.append(os.path.join(label_dir, file))

    return image_paths, label_paths


def make_csv(root):
    file_paths, label_paths = make_file_path(root)

    path_arr = pd.DataFrame({'Image': file_paths, 'Label': label_paths})
    path_arr.to_csv(os.path.join(root, 'file_path.csv'), index=False)


if __name__ == "__main__":
    print("make_csv.py")
    root = '/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/source_labeled'
    make_csv(root)