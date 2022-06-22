import pandas as pd

from doc import Doc
import lxml.etree as ET
import xmltodict

from os import walk


DEF_TWEETS = "twcs.csv"
MAX_TWEETS = 100_000
POST = "POST"
BODY = "BODY"
PREDATOR = "PREDATOR"
CHATLOG = "CHATLOG"
SCREENNAME = "SCREENNAME"
USERNAME = "USERNAME"
VICTIM = "VICTIM"
TEXT = "#text"
PATH = "GeneralData"



def fetch_all_usernames(side_data):
    res = []
    screenames = side_data[SCREENNAME]
    if type(screenames) == list:
        res += [name[USERNAME] for name in screenames]
    else:
        res.append(screenames[USERNAME])
    return res


def get_side_usernames(data, side):
    res = []
    if side not in [PREDATOR, VICTIM]:
        return res

    side_data = data[CHATLOG][side]
    if type(side_data) == list:
        for person in side_data:
            res += fetch_all_usernames(person)
    else:
        res = fetch_all_usernames(side_data)

    return res


def get_predator_usernames(data):
    return get_side_usernames(data, PREDATOR)


def get_victim_usernames(data):
    return get_side_usernames(data, VICTIM)


def get_user_messages(data, user):
    posts = data[CHATLOG][POST]
    resp = []
    for p in posts:
        if p[USERNAME] == user:
            bod = p[BODY]
            if type(bod) == dict:
                if TEXT in bod:
                    resp.append(bod[TEXT])
            else:
                if bod:
                    resp.append(bod)
    return resp


def get_mult_users_messages(data, users):
    res = []
    for user in users:
        res += get_user_messages(data,user)
    return res



def parse_xml(filepath=PATH):
    docs = []
    lists = []
    parser = ET.XMLParser(encoding="utf-8", recover=True)
    filenames = next(walk(filepath), (None, None, []))[2]  # [] if no file

    for f in filenames:
        if f.endswith("xml"):
            try:
                data = xmltodict.parse(ET.tostring(ET.parse(f"{PATH}/{f}", parser=parser).getroot()))

                a, b = get_predator_usernames(data), get_victim_usernames(data)
                for author in a:
                    for message in get_user_messages(data, author):

                        d = Doc(author, True, message)
                        docs.append(d)
                        lists.append(d.to_list())
                for author in b:
                    for message in get_user_messages(data, author):

                        d = Doc(author, False, message)
                        docs.append(d)
                        lists.append(d.to_list())
                # print(a, b)
                # break
            except Exception as e: pass
    return docs, lists


def parse_tweets(csv_filepath=DEF_TWEETS):
    csvv = pd.read_csv(csv_filepath)
    docs, lists = [], []

    for cs in csvv['text'][:MAX_TWEETS]:
        try:
            first_space = cs.index(' ')

            auth = cs[:first_space].replace('@', '')
            text = cs[first_space + 1: ]

            d = Doc(auth, False, text)
            docs.append(d)
            lists.append(d.to_list())
        except: pass

    return docs, lists
