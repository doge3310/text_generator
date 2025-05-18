"""Generate text
"""
import torch
import regex as re
import AI_init as ai
import AI_learn
import text_init


def generate(start_text: str):
    """generate teext from start text

    Args:
        start_text (str): start text for generate

    Returns:
        tuple:
            - (str): generated text
            - (int): lenth of generated text
    """
    start_text = re.sub(r'[^\pL\p{Space}]', '', start_text)
    start_text = start_text.lower().replace("/n", "").split(" ")
    start_text = [text_init.learn_dict.index(i) for i in start_text]

    with torch.no_grad():
        output = ai.transformer.generate(src=start_text)
        print(output)
        text = [text_init.learn_dict[item.tolist()] for item in output]

    return " ".join(text)


if __name__ == "__main__":
    print(generate("в результате пользователь сможет получать"))
