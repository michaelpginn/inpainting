import json
import pathlib
from pprint import pformat

from data.types import Dialog

folder = pathlib.Path(__file__).parent


def load():
    print("Loading orquac")
    data: dict[str, list[Dialog]] = {}

    for split in ["train", "dev", "test"]:
        data[split] = []
        with open(folder / f"{split}.txt", "r") as f:
            # We only want the last turn for each dialog, which has a full history
            current_example = {"id": None, "turns": None}
            for line in f:
                example = json.loads(line)

                # Flush out the last example if we've moved on
                qid = example["qid"].split("#")[0]  # Get rid of the turn number
                if qid != current_example["id"] and current_example["id"] is not None:
                    data[split].append(current_example)  # type:ignore

                turns = [
                    {"question": turn["question"], "response": turn["answer"]["text"]}
                    for turn in example["history"]
                ]
                turns.append(
                    {
                        "question": example["question"],
                        "response": example["answer"]["text"],
                    }
                )
                current_example = {"id": qid, "turns": turns}

        def trim_unanswered(dialog: Dialog):
            for idx, turn in enumerate(dialog["turns"]):
                if turn["response"] == "CANNOTANSWER":
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

    data["test"] = data["dev"] + data["test"]
    del data["dev"]
    print(
        f"Loaded data with splits: {pformat({split: len(dialogs) for split, dialogs in data.items()})}"
    )
    print(f"First example: {pformat(data['train'][0])}")

    return data


if __name__ == "__main__":
    load()
