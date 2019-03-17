"""Prepare experimental data
"""
import re
import csv
import pandas as pd
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split


correct_mapping = {
    "ship": "vận chuyển",
    "shop": "cửa hàng",
    "m": "mình",
    "mik": "mình",
    "ko": "không",
    "k": " không ",
    "kh": "không",
    "khong": "không",
    "kg": "không",
    "khg": "không",
    "tl": "trả lời",
    "r": "rồi",
    "fb": "mạng xã hội", # facebook
    "face": "mạng xã hội",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn",
    "tk": "cảm ơn",
    "ok": "tốt",
    "dc": "được",
    "vs": "với",
    "đt": "điện thoại",
    "thjk": "thích",
    "qá": "quá",
    "trể": "trễ",
    "bgjo": "bao giờ"
}


def tokmap(tok):
    if tok.lower() in correct_mapping:
        return correct_mapping[tok.lower()]
    else:
        return tok


def preprocess(review):
    tokens = review.split()
    tokens = map(tokmap, tokens)
    return " ".join(tokens)


def load_data(filepath, is_train=True):
    regex = 'train_'
    if not is_train:
        regex = 'test_'

    a = []
    b = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if regex in line:
                b.append(a)
                a = [line]
            elif line != "":
                a.append(line)

        b.append(a)

    b = b[1:]
    lst = []
    for tp in b:
        idx = tp[0]
        if is_train:
            lb = int(tp.pop(-1))
        else:
            lb = "0"
        review = " ".join(tp[1:])
        review = re.sub(r"^\"*", "", review)
        review = re.sub(r"\"*$", "", review)
        review_ = preprocess(review)
        review_ws = word_tokenize(review_, format="text")
        lst.append([idx, review, review_ws, lb])
    return lst


if __name__ == "__main__":
    TRAIN_FILE = "./data/train.crash"
    TEST_FILE = "./data/test.crash"

    TRAIN_CSV = "./data/train.csv"
    TEST_CSV = "./data/test.csv"

    DEV_TRAIN_CSV = "./data/train_dev.csv"
    VAL_CSV = "./data/test_dev.csv"

    train_data = load_data(TRAIN_FILE)
    test_data = load_data(TEST_FILE, is_train=False)

    print("# Loaded training samples: {}".format(len(train_data)))
    print("# Loaded test samples: {}".format(len(test_data)))

    cols = ["id", "text", "text_ws", "label"]
    df_train = pd.DataFrame(data=train_data, columns=cols)
    df_train.to_csv(TRAIN_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_test = pd.DataFrame(data=test_data, columns=cols)
    df_test.to_csv(TEST_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)

    train_labels = [tp[-1] for tp in train_data]
    dev_train_data, val_data, _, _ = train_test_split(train_data,
                                                      train_labels, test_size=0.3,
                                                      random_state=42)
    df_dev_train = pd.DataFrame(data=dev_train_data, columns=cols)
    df_val_data = pd.DataFrame(data=val_data, columns=cols)

    df_dev_train.to_csv(DEV_TRAIN_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_val_data.to_csv(VAL_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)

