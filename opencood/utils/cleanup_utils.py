import glob
import os
import sys

def clean_all_numeric_checkpoint(path):
    """
    remove all intermediate checkpoint except bestval

    path: str,
        a path to log directory
    """
    file_list = glob.glob(os.path.join(path, "net_epoch[0-9]*.pth"))
    for file in file_list:
        os.remove(file)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.isdir(path)
    clean_all_numeric_checkpoint(path)