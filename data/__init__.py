from data.orquac.load import load as load_orquac
from data.qrecc.load import load as load_qrecc


def load_dataset():
    orquac = load_orquac()
    qrecc = load_qrecc()

    return {
        "train": orquac["train"] + qrecc["train"],
        "test": orquac["test"] + qrecc["test"],
    }
