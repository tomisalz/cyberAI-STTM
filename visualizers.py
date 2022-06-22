from wordcloud import WordCloud
import matplotlib.pyplot as plt

from script import GSDMM


def create_word_cloud(word_dict):


    wordcloud = WordCloud(background_color='#fcf2ed',
                                width=1800,
                                height=700,

                                colormap='flag').generate_from_frequencies(word_dict)

    # Print to screen
    fig, ax = plt.subplots(figsize=[20,10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

