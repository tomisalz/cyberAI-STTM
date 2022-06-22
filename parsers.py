import time

import pandas as pd

from doc import Doc
import lxml.etree as ET
import xmltodict

from os import walk


DEF_TWEETS = "tweets.csv"
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
    """
    helper function for get_side_usernames
    :param side_data: data of either vitim or predator
    :return:
    """
    res = []
    screenames = side_data[SCREENNAME]
    if type(screenames) == list:
        res += [name[USERNAME] for name in screenames]
    else:
        res.append(screenames[USERNAME])
    return res


def get_side_usernames(data, side):
    """
    parse and find usernames used by either given side (predator\victim)
    :param data: xml data converted to dicts
    :param side: predator or victim
    :return: usernames of side
    """
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
    """
    find all names of predator in data
    :param data: xml data converted to dicts
    :return: usernames used by predator
    """
    return get_side_usernames(data, PREDATOR)


def get_victim_usernames(data):
    """
    find all names of victims in data
    :param data: xml data converted to dicts
    :return: usernames used by victim
    """
    return get_side_usernames(data, VICTIM)


def get_user_messages(data, user):
    """
    returns messages of a given user
    :param data: xml data converted to dicts
    :param user: user to find messages of
    :return: messages of user, as strings
    """
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
    """
    Get messages sent by different users
    :param data: xml data converted to dicts
    :param users: users to find messages of
    :return: messages of users, as strings
    """
    res = []
    for user in users:
        res += get_user_messages(data,user)
    return res



def parse_xml(filepath=PATH):
    """
    Parses xml of baseline messages (chats with predators)
    :param filepath: filepath to xml file
    :return: doc objects of each message
    """
    docs = []
    lists = []
    parser = ET.XMLParser(encoding="utf-8", recover=True)
    filenames = next(walk(filepath), (None, None, []))[2]  # [] if no file
    average = 0
    for idx,f in enumerate(filenames):
        if f.endswith("xml"):
            try:

                data = xmltodict.parse(ET.tostring(ET.parse(f"{PATH}/{f}", parser=parser).getroot()))

                a, b = get_predator_usernames(data), get_victim_usernames(data)
                for author in a:
                    for message in get_user_messages(data, author):
                        start = time.time()
                        d = Doc(author, True, message)
                        average += time.time() - start
                        docs.append(d)
                        lists.append(d.to_list())
                for author in b:
                    for message in get_user_messages(data, author):
                        start = time.time()
                        d = Doc(author, False, message)
                        average += time.time() - start

                        docs.append(d)
                        lists.append(d.to_list())

            except Exception as e: pass

            if idx % 10 == 0:
                print(f"Done parsing {idx + 1} out of {len(filenames)}")
    average = average / len(lists)
    print(f"finished parsing all chats! Average processing time for message: {average}")

    return docs, lists


def parse_tweets(csv_filepath=DEF_TWEETS):
    """
    Parses tweets from datasets into docs
    :param csv_filepath: filepath to csv of tweets
    """
    csvv = pd.read_csv(csv_filepath, encoding='latin-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
    docs, lists = [], []
    average = 0
    for idx, (text,user) in enumerate(zip(csvv['text'][:MAX_TWEETS], csvv['user'][:MAX_TWEETS])):

        try:
            start = time.time()
            d = Doc(user, False, text)
            average += time.time() - start
            docs.append(d)
            lists.append(d.to_list())

        except: pass

        if idx % 1000 == 0:
            print(f"Done parsing {idx + 1} out of {len(csvv[:MAX_TWEETS])}")
    print(f"finished parsing all tweets! Average time per tweet is {average / len(lists)}")
    return docs, lists

