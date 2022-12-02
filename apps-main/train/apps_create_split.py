import json
import os
import pathlib


def create_split(split="train", name="train", difficulty=None):
    """
    update：可以根据难度筛选数据
    """
    paths = []
    roots = sorted(os.listdir(split))
    if difficulty is None:
        for folder in roots:
            root_path = os.path.join(split, folder)
            paths.append(root_path)
    else:
        for folder in roots:
            root_path = os.path.join(split, folder)
            meta_path = os.path.join(root_path, 'metadata.json')
            with open(meta_path, "r") as meta_file:
                meta_data = json.load(meta_file)
                if meta_data['difficulty'] == difficulty:
                    paths.append(root_path)

    file_name = (name + ".json") if difficulty is None else (name + f"_{difficulty}.json")
    with open(file_name, "w") as split_file:
        json.dump(paths, split_file)

    print("file split is done")
    return paths


def create_test_split_by_difficulty():
    for d in difficulty_list:
        create_split(split=paths_to_probs[1], name=names[1], difficulty=d)


# insert path to train and test
# path should be relative to root directory or absolute paths
paths_to_probs = ["../../APPS/train", "../../APPS/test"]
names = ["train", "test"]
difficulty_list = ["introductory", "interview", "competition"]

# all_paths = []
# for index in range(len(paths_to_probs)):
#     all_paths.extend(create_split(split=paths_to_probs[index], name=names[index]))
#
# with open("train_and_test.json", "w") as f:
#     print(f"Writing all paths. Length = {len(all_paths)}")
#     json.dump(all_paths, f)

create_test_split_by_difficulty()
