# -*- coding: utf-8 -*-
# file: absa_utils.py
# time: 02/11/2022 18:55
# author: YANG, HENG
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import copy
import os

import findfile

from pyabsa.framework.flag_class.flag_template import LabelPaddingOption, TaskCodeOption
from pyabsa.tasks.AspectTermExtraction.dataset_utils.__lcf__.atepc_utils import (
    simple_split_text,
)
from pyabsa.utils.data_utils.dataset_item import DatasetItem
from pyabsa.utils.pyabsa_utils import fprint


def generate_inference_set_for_apc(dataset_path):
    """
    Generate inference set for APC dataset. This function only works for APC datasets located in integrated_datasets.
    """
    # Print a message for the user to ensure that the dataset is located in integrated_datasets
    fprint(
        "To ensure your generation is successful, make sure your dataset is located in integrated_datasets."
    )
    # If dataset_path is a DatasetItem object, get the dataset_name attribute
    if isinstance(dataset_path, DatasetItem):
        dataset_name = dataset_path.dataset_name
    else:
        dataset_name = dataset_path

    # Find the train, valid, and test datasets for the specified dataset_name
    train_datasets = findfile.find_cwd_files(
        [
            "dataset",
            "train",
            TaskCodeOption.Aspect_Polarity_Classification,
            dataset_name,
        ],
        exclude_key=[".inference", "readme"],
    )
    valid_datasets = findfile.find_cwd_files(
        [
            "dataset",
            "valid",
            TaskCodeOption.Aspect_Polarity_Classification,
            dataset_name,
        ],
        exclude_key=[".inference", "readme"],
    )
    test_datasets = findfile.find_cwd_files(
        [
            "dataset",
            "test",
            TaskCodeOption.Aspect_Polarity_Classification,
            dataset_name,
        ],
        exclude_key=[".inference", "readme"],
    )
    # Loop through each dataset file and generate an inference set for each one
    for file in train_datasets + valid_datasets + test_datasets:
        try:
            # Open the dataset file and read the lines
            fin = open(file, "r", newline="\n", encoding="utf-8")
            lines = fin.readlines()
            # Loop through each line in the file and check for empty lines
            for i, line in enumerate(lines):
                if not line.strip():
                    raise ValueError(
                        "empty line: #{}, previous line: {}".format(i, lines[i - 1])
                    )
            fin.close()
            # Create the path to save the inference set
            path_to_save = file + ".inference"
            # Open the output file and write the converted lines to it
            fout = open(
                path_to_save, "w", encoding="utf-8", newline="\n", errors="ignore"
            )

            for i in range(0, len(lines), 3):
                # Replace the aspect term with [B-ASP] and [E-ASP] tags
                sample = (
                    lines[i]
                    .strip()
                    .replace("$T$", "[B-ASP]{}[E-ASP]".format(lines[i + 1].strip()))
                )
                # Write the sample and label to the output file
                fout.write(sample + " $LABEL$ " + lines[i + 2].strip() + "\n")
            fout.close()
            # Print the path to the saved inference set
            fprint("save in: {}".format(path_to_save))
        except:
            fprint("Unprocessed file:", file)
    # Print a message to indicate that the inference set generation has finished
    fprint("Inference set generation finished")


def is_similar(s1: str, s2: str) -> bool:
    """
    Determines if two strings are similar based on the number of common tokens they share.
    :param s1: string 1
    :param s2: string 2
    :return: True if strings are similar, False otherwise
    """
    # Split the strings into sets of tokens
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())

    # Compute the intersection of the sets to get common tokens
    common_tokens = tokens1.intersection(tokens2)

    # Compute the similarity as the ratio of common tokens to the maximum length of the two sets
    similarity = len(common_tokens) / max(len(tokens1), len(tokens2))

    # Return True if the similarity is greater than or equal to 0.8, False otherwise
    return similarity >= 0.8


def assemble_aspects(fname, use_tokenizer=False):
    """
    Preprocesses the input file, groups sentences with similar aspects, and generates samples with the corresponding aspect labels and polarities.

    :param fname: The filename to be preprocessed
    :type fname: str
    :param use_tokenizer: Whether to use a tokenizer, defaults to False
    :type use_tokenizer: bool, optional
    :return: A list of samples
    :rtype: list
    """
    # Import tokenizer from transformers library if `use_tokenizer` is True
    if use_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Open and read the input file
    fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    lines = fin.readlines()
    fin.close()

    # Raise an error if an empty line is found
    for i, line in enumerate(lines):
        if not line.strip():
            raise ValueError(
                "empty line: #{}, previous line: {}".format(i, lines[i - 1])
            )

    # Preprocess the data by replacing tokens and splitting the text into tokens
    for i in range(len(lines)):
        if i % 3 == 0 or i % 3 == 1:
            if use_tokenizer:
                lines[i] = (
                    " ".join(tokenizer.tokenize(lines[i].strip()))
                    .replace("$ t $", "$T$")
                    .replace("$ T $", "$T$")
                )
            else:
                lines[i] = (
                    " ".join(simple_split_text(lines[i].strip()))
                    .replace("$ t $", "$T$")
                    .replace("$ T $", "$T$")
                )
        else:
            lines[i] = lines[i].strip()

    # Group sentences with similar aspects and generate samples with the corresponding aspect labels and polarities
    def unify_same_samples(same_samples):
        text = same_samples[0][0].replace("$T$", same_samples[0][1])
        polarities = [LabelPaddingOption.SENTIMENT_PADDING] * len(text.split())
        tags = ["O"] * len(text.split())
        samples = []
        for sample in same_samples:
            polarities_tmp = copy.deepcopy(polarities)

            try:
                asp_begin = sample[0].split().index("$T$")
                asp_end = sample[0].split().index("$T$") + len(sample[1].split())
                for i in range(asp_begin, asp_end):
                    polarities_tmp[i] = sample[2]
                    if i - sample[0].split().index("$T$") < 1:
                        tags[i] = "B-ASP"
                    else:
                        tags[i] = "I-ASP"
                samples.append([text, tags, polarities_tmp])
            except:
                fprint("Ignore Error:", sample[0])

        return samples

    samples = []
    aspects_in_one_sentence = []
    for i in range(0, len(lines), 3):
        lines[i] = lines[i].replace("$T$", " $T$ ").replace("  ", " ")

        if len(aspects_in_one_sentence) == 0:
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
            continue
        if is_similar(aspects_in_one_sentence[-1][0], lines[i]):
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
        else:
            samples.extend(unify_same_samples(aspects_in_one_sentence))
            aspects_in_one_sentence = []
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
    samples.extend(unify_same_samples(aspects_in_one_sentence))

    return samples


def split_aspects(sentence):
    """
    Splits a sentence into multiple aspects, each with its own context and polarity.
    :param sentence: input sentence with multiple aspects
    :return: list of tuples containing single aspect with its context and polarity
    """
    single_aspect_with_contex = []

    aspect_num = len(sentence[1].split("|"))
    aspects = sentence[1].split("|")
    polarity = sentence[2].split("|")
    pre_position = 0
    aspect_context = sentence[0]
    for i in range(aspect_num):
        aspect_context = aspect_context.replace("$A$", aspects[i], 1)
        single_aspect_with_contex.append(
            (
                aspect_context[pre_position: aspect_context.find("$A$")],
                aspects[i],
                polarity[i],
            )
        )
        pre_position = aspect_context.find(aspects[i]) + len(aspects[i]) + 1

    return single_aspect_with_contex


def convert_atepc(fname, use_tokenizer):
    """
    Converts the input file to the Aspect Term Extraction and Polarity Classification (ATEPC) format.
    :param fname: filename
    :param use_tokenizer: whether to use a tokenizer
    """

    fprint("coverting {} to {}.atepc".format(fname, fname))
    dist_fname = fname.replace("apc_datasets", "atepc_datasets")

    if not os.path.exists(os.path.dirname(dist_fname)) and not os.path.isfile(
            dist_fname
    ):
        os.makedirs(os.path.dirname(dist_fname))
    dist_fname += ".atepc"
    lines = []
    samples = assemble_aspects(fname, use_tokenizer)

    for sample in samples:
        for token_index in range(len(sample[1])):
            token, label, polarity = (
                sample[0].split()[token_index],
                sample[1][token_index],
                sample[2][token_index],
            )
            lines.append(token + " " + label + " " + str(polarity))
        lines.append("\n")

    fout = open(dist_fname, "w", encoding="utf8")
    for line in lines:
        fout.writelines((line + "\n").replace("\n\n", "\n"))
    fout.close()


def convert_apc_set_to_atepc_set(path, use_tokenizer=False):
    """
    Converts APC dataset to ATEPC dataset.
    :param path: path to the dataset
    :param use_tokenizer: whether to use a tokenizer
    """
    fprint(
        'To ensure your conversion is successful, make sure the dataset name contain "apc" and "dataset" string '
    )

    if isinstance(path, DatasetItem):
        path = path.dataset_name
    if os.path.isfile(path):
        files = [path]
    elif os.path.exists(path):
        files = findfile.find_files(
            path,
            ["dataset", TaskCodeOption.Aspect_Polarity_Classification],
            exclude_key=[".inference", "readme"],
        )
    else:
        files = findfile.find_cwd_files(
            [path, "dataset", TaskCodeOption.Aspect_Polarity_Classification],
            exclude_key=[".inference", "readme"],
        )

    fprint("Find datasets files at {}:".format(path))
    for target_file in files:
        if not target_file.endswith(".atepc"):
            try:
                convert_atepc(target_file, use_tokenizer)
            except Exception as e:
                fprint("failed to process :{}, Exception: {}".format(target_file, e))
        else:
            fprint("Ignore ", target_file)
    fprint("finished")


def refactor_chinese_dataset(fname, train_fname, test_fname):
    """
    Refactors the Chinese dataset by splitting it into train and test sets and converting it into the ATEPC format.
    :param fname: the name of the dataset file
    :param train_fname: the name of the output train file
    :param test_fname: the name of the output test file
    """
    lines = []
    samples = assemble_aspects(fname)
    positive = 0
    negative = 0
    sum_ = 0
    # refactor testset
    for sample in samples[: int(len(samples) / 5)]:
        for token_index in range(len(sample[1])):
            token, label, polarty = (
                sample[0].split()[token_index],
                sample[1][token_index],
                sample[2][token_index],
            )
            lines.append(token + " " + label + " " + str(polarty))
        lines.append("\n")
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum_ += 1
    fprint(train_fname + f"sum={sum_} positive={positive} negative={negative}")
    if os.path.exists(test_fname):
        os.remove(test_fname)
    fout = open(test_fname, "w", encoding="utf8")
    for line in lines:
        fout.writelines((line + "\n").replace("\n\n", "\n"))
    fout.close()

    positive = 0
    negative = 0
    sum_ = 0
    # refactor trainset
    for sample in samples[int(len(samples) / 5):]:
        for token_index in range(len(sample[1])):
            token, label, polarty = (
                sample[0].split()[token_index],
                sample[1][token_index],
                sample[2][token_index],
            )
            lines.append(token + " " + label + " " + str(polarty))
        lines.append("\n")
        if 1 in sample[2]:
            positive += 1
        else:
            negative += 1
        sum_ += 1
    fprint(train_fname + f"sum={sum_} positive={positive} negative={negative}")
    if os.path.exists(train_fname):
        os.remove(train_fname)
    fout = open(train_fname, "w", encoding="utf8")
    for line in lines:
        fout.writelines((line + "\n").replace("\n\n", "\n"))
    fout.close()


def detect_error_in_dataset(dataset):
    """
    Detects errors in a given dataset by checking if the sentences with similar aspects have different lengths.
    :param dataset: dataset file name
    """
    f = open(dataset, "r", encoding="utf8")
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        # fprint(lines[i].replace('$T$', lines[i + 1].replace('\n', '')))
        if i + 3 < len(lines):
            if is_similar(lines[i], lines[i + 3]) and len(
                    (lines[i] + " " + lines[i + 1]).split()
            ) != len((lines[i + 3] + " " + lines[i + 4]).split()):
                fprint(lines[i].replace("$T$", lines[i + 1].replace("\n", "")))
                fprint(lines[i + 3].replace("$T$", lines[i + 4].replace("\n", "")))
