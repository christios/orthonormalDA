from __future__ import absolute_import, division, print_function

import datasets

model_name= 'UBC-NLP/MARBERT'

_DESCRIPTION = """\
TODO
"""

src_train_path = "/local/ccayral/orthonormalDA/data/coda-corpus/beirut_src.txt"
tgt_train_path = "/local/ccayral/orthonormalDA/data/coda-corpus/beirut_tgt.txt"

class ArabicEmpConv(datasets.GeneratorBasedBuilder):
    """ConvAI: A Dataset of Topic-Oriented Human-to-Chatbot Dialogues"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="spelling_correction",
            version=datasets.Version("1.0.0"),
            description="Full training set",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "src": datasets.Value("string"),
                    "tgt": datasets.Value("string")
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"src": src_train_path,
                            "tgt": tgt_train_path},
            ),
        ]

    def _generate_examples(self, src, tgt):
        with open(src) as f_src, open(tgt) as f_tgt:

            for i, row in enumerate(zip(f_src.readlines(), f_tgt.readlines())):
                yield i, {
                    "src": row[0].strip(),
                    "tgt": row[1].strip(),
                }
