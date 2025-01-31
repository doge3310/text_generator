import regex as re
import numpy as np


def continuation(n_gramm: list, three_gramm: list, text_start: str):
    """func for generate next word

    Args:
        n_gramm (list): text, divided into n-grams
        three_gramm (list): text divided into n-grams in numerical
                            representation
        text_start (str): start of text to continue

    Returns:
        list: continue of text
    """
    text_start = text_start.split(" ")[-2:]

    chanses = []
    answer = []

    for index, item in enumerate(n_gramm):
        if item[0] == text_start[0] and item[1] == text_start[1]:
            chanses.append((three_gramm[index][1:], index))

    for index, item in enumerate(three_gramm):
        if item[1] == chanses[0][0][1] and item[0] == chanses[0][0][0]:
            answer.append(n_gramm[index][1])

    return answer


def three_gramm_generator(text: str):
    """return list of information about text

    Args:
        text (str): text for information extraction

    Returns:
        list: embedings (TF of 3-gramm),
              n_gramm (text 3-gramm),
              p(3) (chance of p(word3|word1, word2)),
              conditional_probability (chance of 3-gramm)
    """
    embedings = []
    n_gramm_ = []
    p_3_ = []
    conditional_probability_ = []
    text = re.sub(r'[^\pL\p{Space}]', '', text)
    list_text = text.lower().replace("/n", "").split(" ")

    for index, _ in enumerate(list_text):
        n_gramm = list_text[index: index + 3]

        p_1 = text.count(n_gramm[0]) / len(list_text)
        p_2 = text.count(" ".join(n_gramm[0: 2])) / text.count(n_gramm[0])
        p_3 = (text.count(" ".join(n_gramm)) /
               text.count(" ".join(n_gramm[: 2])))
        conditional_probability = p_1 * p_2 * p_3

        three_gramm = [round(text.count(i) / len(text), 4) for i in n_gramm]

        embedings.append(three_gramm)
        n_gramm_.append(n_gramm)
        p_3_.append(p_3)
        conditional_probability_.append(conditional_probability)

    return embedings, n_gramm_, p_3_, conditional_probability_


def tf_generator(text: str):
    """generate TF for text

    Args:
        text (str): text for extract

    Returns:
        tf (list): list with TF for all text
    """
    text = re.sub(r'[^\pL\p{Space}]', '', text)
    list_text = text.lower().replace("/n", "").split(" ")
    tf = [round(list_text.count(i) / len(list_text), 4) for i in list_text]

    return tf


def idf_generator(documents: str, lenth_text: int):
    """generate IDF for docs

    Args:
        documents (str): text for extract
        lent_text (int): lenth of one document

    Returns:
        (list): list of idf for every word
    """
    doc_list = []
    documents = re.sub(r'[^\pL\p{Space}]', '', documents)
    documents = documents.lower().replace("/n", "").split(" ")

    for step in range(int(len(documents) / lenth_text) + 1):
        document = documents[lenth_text * step: lenth_text * (step + 1)]
        doc_list.append(document)

    for index, text in enumerate(doc_list):
        for ind, item in enumerate(text):
            doc_list[index][ind] = (count_occurrences(doc_list, item) /
                                    len(doc_list))

        doc_list[index] = text

    doc_list = list(filter(None, doc_list))

    return sum(doc_list, [])


def find_closest_number(number: float, lst: list):
    """find closest float in list

    Args:
        number (float): float number for search
        lst (list): list for search

    Returns:
        (float): closest number
    """
    closest = lst[0]
    smallest_diff = abs(number - closest)

    for num in lst:
        diff = abs(number - num)

        if diff < smallest_diff:
            smallest_diff = diff
            closest = num

    return closest


def find_closest_underlist(underlist: list, lst: list[list]):
    """find closest list in list[list]

    Args:
        underlist (list): list to search
        lst (list[list]): matrix to search closest

    Returns:
        (list): closest list in matrix
    """
    closest = lst[0]
    smallest_diff = [closest[i] - underlist[i] for i in range(len(underlist))]

    for i in lst:
        diff = [i[j] - underlist[j] for j in range(len(underlist))]

        if sum(diff) < sum(smallest_diff):
            smallest_diff = diff
            closest = i

    return closest


def count_occurrences(matrix: list[list], target_string: str):
    """count of occurrences in matrix

    Args:
        matrix (list[list]): matrix to search
        target_string (str): target of string to search

    Returns:
        (int): count of occurrences
    """
    count = 0

    for row in matrix:
        if row.count(target_string) >= 1:
            count += 1

    return count


def min_max_normalizer(x: list):
    """normalize data via min-max normalizer

    Args:
        x (list): list of data

    Returns:
        (list): normalized data
    """
    x = np.array(x)
    min_value = np.min(x)
    max_value = np.max(x)

    normalized_data = (x - min_value) / (max_value - min_value + 0.01)

    return normalized_data


def list_to_matrix(lenth: int, lst: list):
    """generate matrix from list

    Args:
        lenth (int): lenth of matrix string
        lst (list): list to generate matrix

    Returns:
        (list[list]): matrix from list
    """
    output = []

    for index in range(int(len(lst) / lenth) + 1):
        output.append(lst[index * lenth: (index + 1) * lenth])

    return output
