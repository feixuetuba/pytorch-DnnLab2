import os


def is_img(fpath):
    fname = os.path.basename(fpath)
    _, suffix = os.path.splitext(fname)
    return suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]


def load_file_list(fpath, root="", supported_suffix=[".jpg", ".png", ".jpeg"], recurse=False):
    ret = []
    if os.path.isdir(fpath):
        dirs = [fpath]
        while len(dirs) != 0:
            d_path = dirs.pop()
            for f in os.listdir(d_path):
                full_path = f"{d_path}/{f}"
                if os.path.isdir(full_path) and recurse:
                    dirs.append(full_path)
                    continue
                if os.path.splitext(f)[-1].lower() not in supported_suffix:
                    continue
                if os.path.isfile(full_path):
                    ret.append(full_path)
    elif os.path.isfile(fpath):
        _, suffix = os.path.splitext(fpath)
        if suffix.lower() == ".txt":
            with open(fpath, "r") as fd:
                for line in fd.readline():
                    line = line.strip()
                    if os.path.splitext(line)[-1].lower() not in supported_suffix:
                        continue
                    ret.append(os.path.join(root, line))
        elif suffix in supported_suffix:
            ret = [fpath]
    return ret
