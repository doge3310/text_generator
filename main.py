"""Generate text
"""
import torch
import regex as re
import text_init


def generate_tokens(src):
    """generate tokens from tokens

    Args:
        src (list): tokens (start text)

    Returns:
        (tensor): tokens (generatet text)
    """
    tgt = src + [0 for _ in range(text_init.TEXT_LENTH - len(src))]
    tgt = torch.tensor(tgt)
    src = tgt.clone()

    parameters = torch.load("parameters.pkl")
    model = torch.load("architecture.pkl", weights_only=False)
    model.load_state_dict(parameters)

    for word_index in range(len(src) - text_init.TEXT_LENTH,
                            text_init.TEXT_LENTH - 1):
        output = model(src,
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
    start_text = re.findall(r"\b\p{L}+\b", start_text.lower())
    start_text = [text_init.learn_dict.index(i) for i in start_text]

    with torch.no_grad():
        output = generate_tokens(start_text)
        text = [text_init.learn_dict[item.tolist()] for item in output]

    return " ".join(text)


if __name__ == "__main__":
    print(generate_text("arxiv для нас"))
