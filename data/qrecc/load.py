import json
import pathlib
from pprint import pformat

from data.types import Dialog

folder = pathlib.Path(__file__).parent


def load():
    print("Loading qrecc")
    data: dict[str, list[Dialog]] = {}

    for split in ["train", "test"]:
        data[split] = []
        with open(folder / f"qrecc_{split}.json", "r") as f:
            all_turns = json.loads(f.read())

        current_example = {"id": None, "turns": []}
        for turn in all_turns:
            # Flush out the last example if we've moved on
            qid = "qrecc_" + str(turn["Conversation_no"])
            if qid != current_example["id"]:
                if current_example["id"] is not None:
                    data[split].append(current_example)  # type:ignore
                current_example = {"id": qid, "turns": []}

            current_example["turns"].append(
                {"question": turn["Question"], "response": turn["Answer"]}
            )

        def trim_unanswered(dialog: Dialog):
            for idx, turn in enumerate(dialog["turns"]):
                if not turn["response"]:
                    dialog["turns"] = dialog["turns"][:idx]
                    if idx == 0:
                        return None
                    return dialog
            return dialog

        prior_len = len(data[split])
        data[split] = [
            d for d in [trim_unanswered(d) for d in data[split]] if d is not None
        ]
        print(f"Filtered {prior_len - len(data[split])} due to unanswered turns")

    print(
        f"Loaded data with splits: {pformat({split: len(dialogs) for split, dialogs in data.items()})}"
    )
    print(f"First example: {pformat(data['train'][0])}")

    return data


if __name__ == "__main__":
    load()
