"""
Traverses the directory tree and prints the number of directories and images (based on file count) in each folder.

TODO: It's assuming that if the directory has a file, it's an image.
So then if you've got a .DS_Store, the image count for that folder is wrong.
"""
import os


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


walk_through_dir("data/pizza_steak_sushi")
