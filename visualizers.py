import json

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from gsdmm import GSDMM


def create_word_cloud(word_dict):
    """
    creates a word cloud
    :param word_dict:
    :return:
    """

    wordcloud = WordCloud(background_color='#fcf2ed',
                                width=1800,
                                height=700,

                                colormap='flag').generate_from_frequencies(word_dict)

    # Print to screen
    fig, ax = plt.subplots(figsize=[20,10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



def plot_pred_across_epochs(pred_lists):
    ra = range(len(pred_lists[0])) # number of epochs
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(bottom=0.15, left=0.2)

    for idx, c in enumerate(pred_lists):

        ax.plot(ra, c)
        ax.text(ra[-1], c[-1], f"cluster {idx}")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Suspicious messages ratio")

    plt.show()


with open("model_new_0.025_0.6_18_30_2.json", "r") as ff:
    js = json.load(ff)
    gsd = GSDMM()

    gsd.import_from_dict(js)






