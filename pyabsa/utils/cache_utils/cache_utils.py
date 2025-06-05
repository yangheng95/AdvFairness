
import os
import shutil

import findfile


def clean():
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")

    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")

    if os.path.exists("source_datasets.backup"):
        shutil.rmtree("source_datasets.backup")

    if os.path.exists("run"):
        shutil.rmtree("run")

    print("Start cleaning...")
    for f in findfile.find_cwd_files(
            or_key=[".zip", ".cache", ".mv", ".json", ".txt"],
            exclude_key="glove",
            recursive=1,
    ):
        os.remove(f)

    print("Cleaned all files in the current directory.")
