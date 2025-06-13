import regex as re
import glob
import torch


def tokenize(text: str, batch_size: int, stride: int, word_freq: int):
    """generate tokenized text and dict

    Args:
        text (str): row text
        batch_size (int): size of batch
        stride (int): lenth of window in Strided Segmentation

    Returns:
        (list):
            - source (list[list]): tokenized, devided by batches text
            - learn_list (list): list of unique words
    """
    clear_text = re.findall(r"\b\p{L}+\b", text.lower())
    learn_list = {}

    for word in clear_text:  # уникальные слова
        try:
            if word:
                learn_list[word] += 1

        except KeyError:
            learn_list[word] = 1

    words_count = list(learn_list.values())
    learn_list = list(learn_list.keys())

    word_indices = [learn_list.index(word) for word in clear_text
                    if words_count[learn_list.index(word)] <= word_freq]

    elements = [word_indices[i: i + batch_size] +
                ([0] * (batch_size - len(word_indices[i: i + batch_size + 1])))
                for i in range(0, len(word_indices), stride)]

    source = [elements[i: i + batch_size + 1]
              for i in range(0, len(elements), batch_size)]

    return source, learn_list


def check_dataset(text: str):
    """generate info about text (counts of words)

    Args:
        text (str): text for data extraction

    Returns:
        (dict): data info
    """
    clear_text = re.sub(r"[^\pL ]", " ", text)
    clear_text = clear_text.lower().split(" ")
    learn_list = list({i for i in clear_text if i})
    data_info = {}

    for i in learn_list:
        data_info[i] = clear_text.count(i)

    return data_info


learning_text = str()

for file_path in glob.glob("./texts" + "**/*.txt")[: 3000]:
    with open(file_path, mode="r", encoding="UTF-8") as file:
        learning_text += " " + file.read()

TEXT_LENTH = 64
STRIDE = 16
FREQ = 40
src, learn_dict = tokenize(learning_text, TEXT_LENTH, STRIDE, FREQ)
