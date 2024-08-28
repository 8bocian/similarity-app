import argparse
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
from sklearn.manifold import TSNE
import heapq


def calculate_products_distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dist = abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)
    # squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    # dist = np.sqrt(squared_dist)
    return dist


def preprocess(df, n_components):
    tfidf = TfidfVectorizer(analyzer='word')
    tsne = TSNE(n_components=n_components, random_state=42)

    textual_embed_cols = [f'textual_embed_{r}' for r in range(n_components)]
    logger.debug("Preprocessing data")
    df = df.replace({np.nan: None})
    df.fillna("", inplace=True)
    logger.debug("Preprocessing data 1/4")
    df['product_name'] = df['product_name'] + ' ' + df['category']
    df['product_name'] = [clean_string(product_name) for product_name in df['product_name'].values]

    logger.debug("Preprocessing data 2/4")
    textual_matrix = tfidf.fit_transform(df['product_name'])

    logger.debug("Preprocessing data 3/4")

    total_textual_embed = tsne.fit_transform(textual_matrix.toarray())

    df[textual_embed_cols] = [row for row in total_textual_embed]

    df['textual_embed'] = [row for row in df[textual_embed_cols].values]
    df = df[['code', 'product_name', 'textual_embed']]
    logger.debug("Preprocessed data")
    return df


def clean_string(string):
    try:
        alphanum_str = re.sub(r'\W+', ' ', string.lower()).strip()
        alpha_str = re.sub('[0-9]', '', alphanum_str)

        def replace_match(match):
            return match.group(1) + ' ' + match.group(3)

        non_single = re.sub('\b[a-zA-Z0-9]\b', replace_match, alpha_str)

        single_spaces = re.sub('\s{2,}', ' ', non_single)
        return single_spaces
    except:
        return ""


if __name__ == "__main__":
    logger = logging.getLogger('my_logger')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("central_file", type=str, help="path to main csv file", default=None)
    parser.add_argument("client_file", type=str, help="path to secondary csv file", default=None)
    parser.add_argument("output_file", type=str, help="path to putput json file", default=None)

    args = parser.parse_args()
    central_data_path = args.central_file
    shop_data_path = args.client_file
    output_file_name = args.output_file

    n_components = 3

    df_main = pd.read_csv(central_data_path, sep=';')
    df_main = df_main[df_main.columns[:3]]

    df_secondary = pd.read_csv(shop_data_path, sep=';')
    df_secondary = df_secondary[df_secondary.columns[:3]]

    df_main.columns = ['code', 'product_name', 'category']
    df_secondary.columns = ['code', 'product_name', 'category']

    df_ = pd.concat([df_main, df_secondary])
    df_ = df_.astype(str)

    df_ = preprocess(df_, n_components=n_components)

    df_main, df_secondary = df_[:len(df_main)], df_[len(df_main):]

    n_matches = 5

    matched_products = []
    matched_products_codes = []
    distances = []

    for idx_secondary, row_secondary in df_secondary.iterrows():
        product_secondary = row_secondary['textual_embed']

        closest_matches = []

        for idx_main, row_main in df_main.iterrows():
            product_main = row_main['textual_embed']
            distance = calculate_products_distance(p1=product_secondary, p2=product_main)

            heapq.heappush(closest_matches, (-distance, row_main['product_name'], row_main['code']))

            if len(closest_matches) > n_matches:
                heapq.heappop(closest_matches)

        closest_matches.sort()
        matched_products.append([match[1] for match in closest_matches])
        matched_products_codes.append([match[2] for match in closest_matches])
        distances.append([-match[0] for match in closest_matches])

    df_secondary['central_product_matched'] = matched_products
    df_secondary['central_code_matched'] = matched_products_codes
    df_secondary['score'] = distances

    df_secondary = df_secondary.sort_values('score', ascending=True)
    df = df_secondary[['code', 'product_name', 'central_product_matched', 'central_code_matched']]

    df.columns = ['kod_klient', 'nazwa_klient', 'dopasowana_nazwa_centrala', 'dopasowany_kod_centrala']

    df.to_json(output_file_name, orient="records", date_format="iso", date_unit="s")
    logger.debug(f"Synchronized all products")
