import os


def replace_space(directory: str, replace_token: str = '_'):
    """
    Replaces space present in all files/subdirectories with replace_token.
    Note that it does not touch nested directories.
    """
    for fname in os.listdir(directory):
        new_fname = fname.replace(" ", replace_token)
        if new_fname == fname:
            continue
        if os.path.exists(os.path.join(directory, new_fname)):
            print(new_fname, 'exists in the directory. Please delete it before proceeding')
            print('Aborting')
            return
    for fname in os.listdir(directory):
        new_fname = fname.replace(" ", replace_token)
        src = os.path.join(directory, fname)
        dst = os.path.join(directory, new_fname)
        os.rename(src, dst)
        print(src, '--->', dst)
