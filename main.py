import argparse
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
import re
from sklearn.manifold import TSNE



def calculate_products_distance(p1, p2):
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist

def preprocess(df1, df2, n_components):
    textual_embed_cols = [f'textual_embed_{r}' for r in range(n_components)]
    dfs = [df1, df2]
    df = pd.DataFrame()
    for df_ in dfs:
        df = pd.concat([df, df_])
    logger.debug("Preprocessing data")
    df = df.replace({np.nan: None})
    df.fillna("", inplace=True)
    logger.debug("Preprocessing data 1/4")
    df['product_name'] = [clean_string(product_name) for product_name in df['product_name'].values]

    logger.debug("Preprocessing data 2/4")
    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    textual_matrix = tfidf_vectorizer.fit_transform(df['product_name'])

    logger.debug("Preprocessing data 3/4")
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne.get_params()
    total_textual_embed = tsne.fit_transform(textual_matrix.toarray())
    df[textual_embed_cols] = [row for row in total_textual_embed]

    df['textual_embed'] = [row for row in df[textual_embed_cols].values]
    df = df[['code', 'product_name', 'textual_embed']]
    logger.debug("Preprocessing data 4/4")
    return df[:len(df1)], df[len(df1):]


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


def project(df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    field = "textual_embed"

    m, zlow, zhigh = ('^', -30, -5)
    xs = [row[0] for row in df[field].values]
    ys = [row[1] for row in df[field].values]
    zs = [row[2] for row in df[field].values]

    ax.scatter(xs, ys, zs, marker=m)

    ax.set_xlabel(field + "0")
    ax.set_ylabel(field + "1")
    ax.set_zlabel(field + "2")

    plt.show()


if __name__ == "__main__":
    dotenv.load_dotenv()

    logger = logging.getLogger('my_logger')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--main_csv", type=str, help="path to main csv file", default=None)
    parser.add_argument("--secondary_csv", type=str, help="path to secondary csv file", default=None)

    args = parser.parse_args()
    main_csv_path = args.main_csv
    secondary_csv_path = args.secondary_csv

    n_components = 3

    df_main = pd.read_csv(main_csv_path, sep=';')
    df_main = df_main[df_main.columns[:2]]

    df_secondary = pd.read_csv(secondary_csv_path, sep=';')
    df_secondary = df_secondary[df_secondary.columns[:2]]

    # df_main = df_main.iloc[:20]
    # df_secondary = df_secondary.iloc[:20]

    df_main.columns = ['code', 'product_name']
    df_secondary.columns = ['code', 'product_name']
    df_main_len = len(df_main)
    df_main, df_secondary = preprocess(df_main, df_secondary, n_components=n_components)

    n_matches = 1
    matched_products = []
    distances = []
    print(len(df_main), len(df_secondary))
    for idx_secondary, row_secondary in df_secondary.iterrows():
        product_secondary = row_secondary['textual_embed']
        min_dist = np.inf
        matched_product = None

        for idx_main, row_main in df_main.iterrows():
            product_main = row_main['textual_embed']
            distance = calculate_products_distance(p1=product_secondary, p2=product_main)
            if distance < min_dist:
                min_dist = distance
                matched_product = row_main['product_name']
        matched_products.append(matched_product)
        distances.append(min_dist)
    df_secondary['central_product_matched'] = matched_products
    df_secondary['score'] = distances
    df_secondary = df_secondary.sort_values('score', ascending=True)
    df = df_secondary[['code', 'product_name', 'central_product_matched', 'score']]
    df.to_csv(f'result_pairing_clean_lubanska_4343.csv')
    logger.debug(f"Synchronized all products")
