"""
multilingual universal sentence encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader

hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")
tensorflow_text = LazyLoader("tensorflow_text", globals(), "tensorflow_text")


class MultilingualUniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Multilingual Universal
    Sentence Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        tensorflow_text._load()
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
            mirror_tfhub_url = "https://hub.tensorflow.google.cn/google/universal-sentence-encoder-multilingual-large/3"
        else:
            tfhub_url = "https://https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            mirror_tfhub_url = "https://hub.tensorflow.google.cn/google/universal-sentence-encoder-multilingual/3"

        # TODO add QA SET. Details at: https://hub.tensorflow.google.cn/google/universal-sentence-encoder-multilingual-qa/3
        self._tfhub_url = tfhub_url
        self.mirror_tfhub_url = mirror_tfhub_url
        try:
            self.model = hub.load(self._tfhub_url)
        except Exception as e:
            print("Error loading model from tfhub, trying mirror url, Exception: ", e)
            self.model = hub.load(self.mirror_tfhub_url)

    def encode(self, sentences):
        return self.model(sentences).numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        try:
            self.model = hub.load(self._tfhub_url)
        except Exception as e:
            print("Error loading model from tfhub, trying mirror url, Exception: ", e)
            self.model = hub.load(self.mirror_tfhub_url)
