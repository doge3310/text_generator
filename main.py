"""Generate text
"""
import torch
import regex as re
import AI_init as ai
import AI_learn
import text_init


def generate_tokens(src):
    tgt = src + [0 for _ in range(text_init.TEXT_LENTH - len(src))]
    tgt = torch.tensor(tgt)
    src = tgt

    for word_index in range(len(src) - text_init.TEXT_LENTH,
                            text_init.TEXT_LENTH - 1):
        output = ai.transformer(src,
                                tgt.long())
        output = torch.argmax(output, dim=-1)
        tgt[word_index] = output[0][word_index]

    return tgt


def generate_text(start_text: str):
    """generate text from start text

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
        output = generate_tokens(start_text)
        text = [text_init.learn_dict[item.tolist()] for item in output]

    return " ".join(text)


if __name__ == "__main__":
    print(generate_text("в результате пользователь сможет получать"))
