import regex as re


text = "как ты ты это делаешь тумба юмба sdg aol;sndg liaundfg ;omdfgj p;[io ksoidfu iukats diol] asdfg"
bs = 3


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
                ([0] * (bs - len(word_indices[i: i + batch_size])))
                for i in range(0, len(word_indices), batch_size)]

    source = [elements[i: i + batch_size]
              for i in range(0, len(elements), batch_size)]

    return source, learn_list


print(tokenize(text, bs)[0])
