"""Generate text
"""
import AI_init as ai


def generate_text(lenth_text: int):
    pass


if __name__ == "__main__":
    print(generate_text(20))


# y_pred = torch.tensor(rlt.three_gramm_l[-1], dtype=torch.float32)
# x = torch.tensor(rlt.three_gramm_l[-2], dtype=torch.float32)
# test_text = []

# print(x, y_pred)

# for index in range(10):
#     x = y_pred
#     y_pred = NN.forward(x)
#     closest_embeding = apps.find_closest_underlist(y_pred.tolist(), rlt.three_gramm_l)
#     closest_index = rlt.three_gramm_l.index(closest_embeding)

#     test_text.append(rlt.n_gramm_l[closest_index])

# print(test_text)
