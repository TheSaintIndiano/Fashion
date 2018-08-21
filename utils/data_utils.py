__author__ = 'indiano'

# encoding=utf8

import fnmatch
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

from config import config


class DataUtils:

    def __init__(self, args):
        self.args = args

    def load_txt_files(self):
        """
        :param root_dir:str
        :return: dataframe

        load_txt_files function loads raw text files and sanitize them.
        """

        # Walking through all the events file
        files = []
        for root, dirnames, filenames in os.walk(os.path.join(self.args.root_dir, self.args.raw_data_dir)):
            for filename in fnmatch.filter(filenames, 'event*.txt'):
                files.append(os.path.join(root, filename))

        # Sanitizing all the event file and collecting them in dataframe
        sanitized_df = pd.DataFrame()
        for file in files:
            santized_flattened_dict = self.filter_data(open(file, "r").read())
            sanitized_df = pd.concat([sanitized_df, pd.DataFrame(santized_flattened_dict)], axis=0, ignore_index=True)

        # Flattening column tags containing dict values and concatenating with others
        sanitized_df['tags'] = sanitized_df['tags'].apply(lambda value: np.nan if len(value) == 0 else value)
        tags = sanitized_df.tags.apply(pd.Series)
        sanitized_df = pd.concat([sanitized_df, tags], axis=1)
        sanitized_df['color'].fillna(sanitized_df['value'], inplace=True)
        sanitized_df.rename(columns={'offer.price.amount': 'amount'}, inplace=True)

        self.save_data(sanitized_df)

    def filter_data(self, data):
        """
        :param data: str
        :return:

        filter_data function filters out relevent string blocks on the basis of regular expression.
        """

        # Regular expression to match the occurence of product ID ie. guid
        regex = re.compile(r'({"guid".+?]})', re.IGNORECASE)
        matches = regex.findall(data)

        # Removing first match as it contains redundent, duplicate guid values
        matches = matches[1:]

        return [self.flatten_dict(json.loads(match)) for match in matches]

    def flatten_dict(self, dictionary):
        """
        :param dictionary: dict
        :return:

        flatten_dict function flattens dataframe columns containing dict
        """

        def expand(key, value):
            if isinstance(value, dict):
                return [(key + '.' + k, v) for k, v in self.flatten_dict(value).items()]
            else:
                return [(key, value)]

        items = [item for k, v in dictionary.items() for item in expand(k, v)]
        return dict(items)

    def filter_url(self, data):
        """
        :param data: str
        :return:

        filter_url function filters out list of url strings on the basis of regular expression.
        """

        # Regular expression to filter list of url strings after https://........../s/ ................ /
        # eg. https://shop.nordstrom.com/s/eliza-j-lace-fit-flare-dress-regular-petite/3651256?fashioncolor=Navy&fashionsize=Regular-4
        # eliza-j-lace-fit-flare-dress-regular-petite
        regex = re.compile(r'/s/([a-zA-Z0-9_-]+)', re.IGNORECASE)
        matches = regex.findall(str(data))

        if len(matches) > 0:
            for str in matches:
                url_str_list = str.split('-')
                url_str_list = (' ').join(url_str_list)
        else:
            url_str_list = np.nan
        return url_str_list

    def label_data(self, df):
        """
        Assigning labels based on regex filtering
        :param df: dataframe
        :return: dataframe
        """

        # Assigning all labels as 0 to nada sportswear and 1 to sportswear after filtering out using pandas contains
        df['label'] = 0
        df['label'].values[df.url.str.lower().str.contains('|'.join(config.sportswear))] = 1

        return df

    def save_data(self, df):
        """
        Saving dataframe in HDF file system format

        :param df: dataframe
        :return:
        """

        df['url'] = pd.Series(self.filter_url(df['url']))
        df.drop_duplicates(['url'], inplace=True)
        df = self.label_data(df)

        # Printing info, statistics, top 5 rows
        print(df.info())
        print(df.describe())
        print(df.head())

        df.to_hdf(self.args.hdf_file + '.hdf', self.args.hdf_file, mode='w', format='fixed')
        print("Dataframe created & saved.")

    def impute_missing_values(self, encoded_x):
        """
        :param encoded_x:np.array
        :return: np.array

        impute_missing_values function impute missing values as the mean or median.
        """
        # Defining & Fitting sklearn Imputer
        imputed_x = Imputer().fit_transform(encoded_x)
        print("Imputed shape: ", imputed_x.shape)
        print("******************* Imputing done ************************")
        return imputed_x

    def prepare_data(self, impute=False, fillbyzero=False):
        """
        Prepare_data function prepares data for train & test set

        :param data_dir:str
        :param hdf_file:str
        :param impute:bool
        :param fillbyzero:bool
        :return: X_train, X_test, y_train, y_test:np.array, np.array, np.array, np.array
        """

        # load data
        data = pd.read_hdf(self.args.hdf_file + '.hdf')
        data = data.loc[:, ['guid', 'xid', 'color', 'size', 'url', 'amount', 'label']]

        # Fill missing values by zero or impute by mean or median at later stage
        if fillbyzero:
            data = data.fillna(0)
        elif impute:
            data = self.impute_missing_values(data)

        # Categorical input features eg. categorical_features = ['url'] or ['color', 'size', 'url']
        df = data.loc[:, ['url', 'label']]
        df.drop_duplicates(subset=['url'], inplace=True)

        return df

    def data_Stats(self, y_train, y_val, y_test):
        # Quantifying the Nada Sportswear & Sportswear in the dataset
        counter = Counter(y_train)
        print('\n\nTrain set entries.')
        for key in counter:
            if key == 0:
                print('{:.2f}% Nada Sportswear'.format((counter[key] / len(y_train)) * 100))
            elif key == 1:
                print('{:.2f}% Sportswear'.format((counter[key] / len(y_train)) * 100))

        counter = Counter(y_val)
        print('\nVal set entries.')
        for key in counter:
            if key == 0:
                print('{:.2f}% Nada Sportswear'.format((counter[key] / len(y_val)) * 100))
            elif key == 1:
                print('{:.2f}% Sportswear'.format((counter[key] / len(y_val)) * 100))

        counter = Counter(y_test)
        print('\nTest set entries.')
        for key in counter:
            if key == 0:
                print('{:.2f}% Nada Sportswear'.format((counter[key] / len(y_test)) * 100))
            elif key == 1:
                print('{:.2f}% Sportswear'.format((counter[key] / len(y_test)) * 100))
