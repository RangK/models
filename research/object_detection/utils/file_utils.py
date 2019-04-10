import os


def get_sub_dirs(root_dir, max_depth=0):
    return get_childs(root_dir, is_dir=True, max_depth=max_depth)


def get_files(root_dir, extension='.jpg', max_depth=0):
    return get_childs(root_dir, is_dir=False, extension=extension, max_depth=max_depth)


def get_childs(root_dir, is_dir=False, extension='.jpg', max_depth=0):
    """
    get files or directories related root dir
    :param root_dir:
    :param is_dir:
    :param extension:
    :param max_depth:
    :return:
    """
    if os.path.exists(root_dir) is False:
        raise FileNotFoundError("not exist dir : {}".format(root_dir))

    target_items = []

    childs, next_dirs = _get_sub_childs(root_dir, is_dir, extension)
    target_items.extend(childs)

    while max_depth > 0:
        next_sub_dirs = []
        for sub in next_dirs:
            if not os.path.isdir(sub):
                continue

            sub_child_items, sub_dirs = _get_sub_childs(sub, is_dir, extension)

            next_sub_dirs.extend(sub_dirs)
            target_items.extend(sub_child_items)

        max_depth -= 1
        next_dirs = next_sub_dirs

    return target_items


def _get_sub_childs(root_path, is_dir=False, extension='.jpg'):
    targets = []
    sub_dirs = []
    subs = os.listdir(root_path)
    for sub in subs:
        sub_path = os.path.join(root_path, sub)
        sub_is_dir = os.path.isdir(sub_path)

        if sub_is_dir:
            sub_dirs.append(sub_path)

        if sub_is_dir is not is_dir:
            continue

        if is_dir is False and sub_path.endswith(extension) is False:
            continue

        targets.append(sub_path)

    return targets, sub_dirs


def get_lines_by_file(file):
    file.seek(0)

    return [line.strip() for line in file]


def get_lines(file_path):
    file = open(file_path, "r", encoding="utf-8")

    return get_lines_by_file(file)


def get_last_name_from_path(full_path):
    paths = full_path.split("/")
    return paths[len(paths) - 1] if len(paths) > 0 else ""


def makedirs_ifnot(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

        return True

    return False


def open_writable_file(path, mode="w+"):
    return open(path, mode, encoding='utf-8')
