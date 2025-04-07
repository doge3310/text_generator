import regex as re


file = open("./AI.txt", "r", encoding="utf-8")
learning_text = file.read()
TEST_TEXT = """Чтобы понять, почему мы это делаем"""
TEXT_LENTH = 60

learning_text = re.sub(r'[^\pL\p{Space}]', '', learning_text)
learning_text = learning_text.lower().replace("/n", "").split(" ")
learn_list = list(set(learning_text))
index = 0
src = [[] for _ in range(int(len(learning_text) / TEXT_LENTH))]

for w_index, word in enumerate(learning_text):
    if w_index % TEXT_LENTH == 0 and w_index != 0:
        index += 1

    src[index].append([learn_list.index(item) for item in
                       learning_text[TEXT_LENTH * index: TEXT_LENTH * (index + 1)]])
