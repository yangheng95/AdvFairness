import os

import pandas as pd
import pinyin

from . import WordSwap


class ChineseHomophoneCharacterSwap(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by a
    homophone dictionary."""

    def __init__(self):
        # Get the absolute path of the homophone dictionary txt
        path = os.path.dirname(os.path.abspath(__file__))
        path_list = path.split(os.sep)
        path_list = path_list[:-2]
        path_list.append("shared/chinese_homophone_char.txt")
        homophone_tuple_path = os.path.join("/", *path_list)
        homophone_tuple = pd.read_csv(homophone_tuple_path, header=None, sep="\n")
        homophone_tuple = homophone_tuple[0].str.split("\t", expand=True)
        self.homophone_tuple = homophone_tuple

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with 1 character
        replaced by a homophone."""
        candidate_words = []
        for i in range(len(word)):
            character = word[i]
            character = pinyin.get(character, format="strip", delimiter=" ")
            if character in self.homophone_tuple.values:
                for row in range(self.homophone_tuple.shape[0]):  # df is the DataFrame
                    for col in range(0, 1):
                        if self.homophone_tuple._get_value(row, col) == character:
                            for j in range(1, 4):
                                repl_character = self.homophone_tuple[col + j][row]
                                if repl_character is None:
                                    break
                                candidate_word = (
                                    word[:i] + repl_character + word[i + 1 :]
                                )
                                candidate_words.append(candidate_word)
            else:
                pass
        return candidate_words
