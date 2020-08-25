import os
import shutil

import pickle


def dirs_creation(dirs, wipe_dir=False):
    """
    """
    for dir in dirs:

        if os.path.isdir(dir) and wipe_dir:
            shutil.rmtree(dir)
            os.mkdir(dir)
        elif not os.path.isdir(dir):
            os.mkdir(dir)
        else:
            print(f'Directory {dir} already exists')
            print(f'wipe_dir is set to {wipe_dir}')

    return None


def dump_pickle(objs, paths, filenames):
    """
    """
    for obj, path, filename in zip(objs, paths, filenames):

        out_pkl = open(
            f'{path}\\{filename}.pkl',
            'wb'
        )
        pickle.dump(obj, out_pkl)
        out_pkl.close()

    return None
