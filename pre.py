import pandas as pd
import ast
import pickle
from sentence_transformers import SentenceTransformer
import os
import random
import stanza
import pickle
from collections import Counter,defaultdict
from sklearn.preprocessing import LabelEncoder


def read_data(file_path, min_freq=1):
    df = pd.read_csv(file_path)

    # Filter out music styles with less than min_freq (2000) appearances
    style_counts = df["music style"].value_counts()
    valid_styles = style_counts[style_counts >= min_freq].index
    df = df[df["music style"].isin(valid_styles)]
    df = df[df["words count"]>9]

    artists = df["artist"].tolist()
    df["songs"] = df["songs"].apply(lambda song_str: " ".join(ast.literal_eval(song_str)))
    songs = df["songs"].tolist()

    # Convert the "music style" column to categorical
    df["music style"] = pd.Categorical(df["music style"])
    style_codes = df["music style"].cat.codes.tolist()
    style_mapping = dict(enumerate(df["music style"].cat.categories))

    print("Music style dictionary: {}".format(style_mapping))

    style_list = df["music style"].tolist()

    # Get statistics about the music styles
    style_stats = df["music style"].value_counts().to_dict()
    print(style_stats)

    return artists, songs, style_codes, style_mapping, df


def encode_data(raw_texts, model_name, target_path):
    if os.path.exists(target_path):
        # Load embedding matrix from file if it exists
        with open(target_path, 'rb') as file:
            embedding_matrix = pickle.load(file)
    else:
        # Encode using SentenceTransformer model and save to file
        model = SentenceTransformer(model_name)
        embedding_matrix = model.encode(raw_texts)

        with open(target_path, 'wb') as file:
            pickle.dump(embedding_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_matrix


def lemmatize_texts(texts, target_path, language='he'):
    if os.path.exists(target_path):
        # Load embedding matrix from file if it exists
        with open(target_path, 'rb') as file:
            lemmatized_texts = pickle.load(file)
    else:
        # Download stanza model for the specified language
        stanza.download(language)

        # Initialize a custom pipeline with only tokenizer and lemmatizer
        nlp = stanza.Pipeline(language, processors='tokenize,lemma')

        lemmatized_texts = []
        i = 0
        for text in texts:
            # Process the text and xtract lemmas for each word in the text
            doc = nlp(text)
            print(i)
            i += 1
            lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
            lemmatized_text = ' '.join(lemmas)
            lemmatized_texts.append(lemmatized_text)

    with open(target_path, 'wb') as file:
        # Save lemmetized data
        pickle.dump(lemmatized_texts, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return lemmatized_texts


def prepare_balanced_author_identification_datasets(df, min_songs_per_artist=2):
    # Group by music style and artist, count the songs
    style_artist_songs = df.groupby(['music style', 'artist'])['songs'].count().reset_index()

    # Rename 'songs' column to 'song_count' for clarity
    style_artist_songs = style_artist_songs.rename(columns={'songs': 'song_count'})

    # Filter artists with at least min_songs_per_artist songs
    style_artist_songs = style_artist_songs[style_artist_songs['song_count'] >= min_songs_per_artist]

    # Count artists per style and heck if we have enough data
    style_artist_counts = style_artist_songs['music style'].value_counts()
    
    if len(style_artist_counts) == 0:
        print("Not enough data to create balanced datasets. Please lower the min_songs_per_artist value.")
        return None

    # Find the minimum number of artists across styles
    min_artists = style_artist_counts.min()

    # Find the minimum number of songs per artist
    min_songs = style_artist_songs['song_count'].min()

    balanced_datasets = {}

    for style in style_artist_counts.index:
        style_data = style_artist_songs[style_artist_songs['music style'] == style]

        # Check if enough artists exist for the style
        if len(style_data) < min_artists:
            print(f"Not enough artists for style {style}. Skipping this style.")
            continue

        # Randomly select min_artists artists
        selected_artists = random.sample(list(style_data['artist']), min_artists)

        style_dataset = []
        for artist in selected_artists:
            # Randomly select min_songs songs
            artist_songs = df[(df['music style'] == style) & (df['artist'] == artist)]['songs'].tolist()
            selected_songs = random.sample(artist_songs, min_songs)
            style_dataset.extend([(song, artist) for song in selected_songs])

        balanced_datasets[style] = style_dataset

    # Print summary
    if balanced_datasets:
        print(f"Balanced datasets created for author identification:")
        print(f"Number of styles: {len(balanced_datasets)}")
        print(f"Artists per style: {min_artists}")
        print(f"Songs per artist: {min_songs}")
        print("\nDataset sizes:")
        for style, dataset in balanced_datasets.items():
            print(f"{style}: {len(dataset)} samples")
    else:
        print("No balanced datasets could be created with the given constraints.")

    return balanced_datasets


def encode_data_fast(raw_texts, model_name):
    model = SentenceTransformer(model_name)
    embedding_matrix = model.encode(raw_texts)
    return embedding_matrix


def prepare_data_for_classification(balanced_datasets, style, model_name):

    if style not in balanced_datasets:
        raise ValueError(f"Style '{style}' not found in balanced_datasets")

    # Extract songs and artists
    songs, artists = zip(*balanced_datasets[style])

    # Encode the songs into a feature matrix
    embeddings = encode_data_fast(list(songs), model_name)

    # Encode the artists to numerical labels (classes)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(artists)
    classes = label_encoder.classes_

    return embeddings, labels, classes
