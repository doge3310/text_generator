import apps
from numpy import array

file = open(".\AI.txt", "r", encoding="utf-8")
learning_text = file.read()
TEST_TEXT = """Чтобы понять, почему мы это делаем"""
TEXT_LENTH = 100

learning_text_tf = apps.tf_generator(learning_text)
learning_text_idf = apps.idf_generator(learning_text, TEXT_LENTH)
learn_tf_idf = apps.min_max_normalizer(array(learning_text_tf) * array(learning_text_idf))
learn_tf_idf = learn_tf_idf.tolist()
learn_text = []

for index in range(int(len(learning_text_tf) / TEXT_LENTH) + 1):
    element: list = learn_tf_idf[TEXT_LENTH * index: TEXT_LENTH * (index + 1)]
    learn_text.append(element)

if __name__ == "__main__":
    for i in learn_text:
        print(i)
