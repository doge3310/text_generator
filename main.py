"""Generate text
"""
import torch
import AI_init as ai
import regex as re
import AI_learn
import text_init


def generate(start_text: str):
    start_text = re.sub(r'[^\pL\p{Space}]', '', start_text)
    start_text = start_text.lower().replace("/n", "").split(" ")
    start_text = [text_init.learn_dict.index(i) for i in start_text]

    with torch.no_grad():
        output = ai.transformer.generate(src=start_text)
        text = [text_init.learn_dict[item.tolist()] for item in output[0]]

    return " ".join(text), len(text)


if __name__ == "__main__":
    print(generate("В результате пользователь сможет получать"))
