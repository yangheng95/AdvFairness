
from pyabsa import TADCheckpointManager

from textattack.model_args import PYABSA_MODELS
from textattack.reactive_defense.reactive_defender import ReactiveDefender


class TADReactiveDefender(ReactiveDefender):
    """Transformers sentiment analysis pipeline returns a list of responses
    like.

        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

    We need to convert that to a format TextAttack understands, like

        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, ckpt="tad-sst2", defense="pwws", **kwargs):
        super().__init__(**kwargs)
        self.defense = defense
        self.tad_classifier = TADCheckpointManager.get_tad_text_classifier(
            checkpoint=PYABSA_MODELS[ckpt], auto_device=True
        )

    def repair(self, text, **kwargs):
        res = self.tad_classifier.infer(
            text, defense=self.defense, print_result=False, **kwargs
        )
        return res
