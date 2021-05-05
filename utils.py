import os
import errno


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def isdir(dirname):
    return os.path.isdir(dirname)
