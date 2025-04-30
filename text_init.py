import regex as re


def tokenize(text: str, batch_size: int):
    """generate tokenized text and dict

    Args:
        text (str): row text
        batch_size (int): size of batch

    Returns:
        (list):
            - source (list[list]): tokenized, devided by batches text
            - learn_list (list): list of unique words
    """
    clear_text = re.sub(r'[^\pL\p{Space}]', '', text)
    clear_text = clear_text.lower().replace("/n", "").split(" ")
    learn_list = list(set(clear_text))

    word_indices = [learn_list.index(word) for word in clear_text]

    elements = [word_indices[i: i + batch_size] +
                ([0] * (batch_size - len(word_indices[i: i + batch_size])))
                for i in range(0, len(word_indices), batch_size)]

    source = [elements[i: i + batch_size]
              for i in range(0, len(elements), batch_size)]

    return source, learn_list


file = open("./AI.txt", "r", encoding="utf-8")
learning_text = file.read()
TEXT_LENTH = 60
src, learn_dict = tokenize(learning_text, TEXT_LENTH)
