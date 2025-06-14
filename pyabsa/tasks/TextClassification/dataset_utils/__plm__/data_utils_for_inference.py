# -*- coding: utf-8 -*-
# file: data_utils_for_inference.py
# time: 02/11/2022 15:39


import tqdm
from torch.utils.data import Dataset

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import fprint


class BERTTCInferenceDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        if isinstance(text, list):
            self.process_data(text, ignore_error=ignore_error)
        else:
            self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):
        lines = load_dataset_from_file(infer_file, config=self.config)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        if len(samples) > 100:
            it = tqdm.tqdm(samples, desc="preparing text classification dataloader")
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference datasets
                if text is None or "" == text.strip():
                    raise RuntimeError("Invalid Input!")

                if "$LABEL$" in text:
                    text, _, label = text.strip().partition("$LABEL$")
                    label = label.strip()
                    text = text.strip().lower()

                else:
                    label = LabelPaddingOption.LABEL_PADDING

                text_indices = self.tokenizer.text_to_sequence("{}".format(text))

                data = {
                    "ex_id": ex_id,
                    "text_indices": text_indices
                    if "text_indices" in self.config.model.inputs
                    else 0,
                    "text_raw": text,
                    "label": label,
                }

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    fprint("Ignore error while processing:", text)
                else:
                    raise e

        self.data = all_data

        self.data = PyABSADataset.covert_to_tensor(self.data)

        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
