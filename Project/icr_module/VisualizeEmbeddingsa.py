from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

EMBEDDING_PICKLE = 'thumbnail_to_embedding.pickle'

sns.set()
import pickle
import glob
import os


def save(something, filename):
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(something, handle)


def load(something):
    with open(something, 'rb') as handle:
        return pickle.load(handle)


def run_pca_on_embeddings(embeddings):
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


def run_tsne_on_embeddings(embeddings):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    return tsne.fit_transform(embeddings)
    # save(tsne_results, 'tsne')


def get_label_list():
    img_dir = "D:\\Data\\thumbnails\\thumbnails\\**\\*.jpg"
    data_path = os.path.join(img_dir)
    files = glob.glob(data_path)
    id_to_thumbnail = load(EMBEDDING_PICKLE)

    file_to_label = {}
    for file in files:
        label = file[:file.rfind('\\')][-2:].replace('\\', '')
        file_id = file[file.rfind("\\") + 1:-4]
        file_to_label[file_id] = int(label)

    labels = []
    thumbnails = []
    for id in id_to_thumbnail.keys():
        label = file_to_label[id]
        if label in LABELS_TO_KEEP:
            labels.append(file_to_label[id])
            thumbnails.append(id_to_thumbnail[id])
        # else:
        #     labels.append(1)

    pca = run_pca_on_embeddings(thumbnails)
    # tsne = run_tsne_on_embeddings(thumbnails)

    return labels, pca


# run_tsne_on_embeddings()
# Counter({2: 39835, 4: 7539, 3: 7440, 5: 6235, 6: 2706, 7: 2507, 11: 1377, 14: 909, 13: 779, 20: 590, 16: 537, 12: 449, 1: 381, 10: 356, 15: 306, 24: 245, 19: 139, 8: 133, 21: 72, 25: 69, 27: 63, 29: 63, 9: 45, 26: 42, 28: 29, 17: 25, 22: 20, 18: 15, 30: 11, 34: 7, 31: 5, 23: 3, 38: 2, 33: 2, 35: 2, 36: 1, 37: 1, 32: 1, 39: 1})
LABELS_TO_KEEP = (11, 20, 12, 10, 15, 2)
# pcas = load('thumbnail_pca.pickle')
labels, pcas = get_label_list()

from collections import Counter
print(Counter(labels))

sns.scatterplot(x=pcas[:, 0], y=pcas[:, 1], alpha=0.5, hue=labels, palette=sns.hls_palette(len(LABELS_TO_KEEP), l=.3, s=.8))
plt.axis('equal')
plt.show()

# tsne = load('tsne.pickle')
plt.scatter(tsne[:, 0], tsne[:, 1], alpha=0.5, c=labels)
plt.axis('equal')
plt.show()
#


