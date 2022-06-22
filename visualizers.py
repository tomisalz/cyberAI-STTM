from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from script import GSDMM


with open("model.json", "r") as mod_f:
    mod = json.load(mod_f)
    gds = GSDMM()
    gds.import_from_dict(mod)


print("here")
# print(gds.clusters[1].items())
# Select topic you want to output as dictionary (using topic_number)
topic_dict = sorted(gds.clusters[1].nwz.items(), key=lambda k: k[1], reverse=True)[:40]
print(gds.clusters[1].stats())
# Generate a word cloud image
wordcloud = WordCloud(background_color='#fcf2ed',
                            width=1800,
                            height=700,

                            colormap='flag').generate_from_frequencies(gds.clusters[1].nwz)

# Print to screen
fig, ax = plt.subplots(figsize=[20,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

