import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import os


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def move_to_label_folders(files):
    BASE_FOLDER = "D:\\Data\\thumbnails\\thumbnails\\"
    id_to_label = \
    pd.read_csv("D:\\Data\\adressa_articles.csv", usecols=['id', 'category0_encoded'], index_col=['id']).to_dict()[
        'category0_encoded']

    for file in files:
        file_id = file[file.rfind("\\") + 1:-4]
        if file_id in id_to_label:
            file_category = str(id_to_label[file_id])
            new_location = BASE_FOLDER + file_category + "\\" + file[file.rfind("\\") + 1:]
            os.makedirs(os.path.dirname(new_location), exist_ok=True)
            shutil.move(file, new_location)
        else:
            continue
