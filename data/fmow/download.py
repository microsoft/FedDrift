# downloads all images (55 GB) to root_dir
# should provide root_dir as the command-line argument when running the main script
from wilds import get_dataset
dataset = get_dataset(dataset="fmow", root_dir="data", download=True)
