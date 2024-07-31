import pandas
import pre
import exp
import train

if __name__ == '__main__':
    model_name = "imvladikon/sentence-transformers-alephbert"
    file_path = "data/kaggle.csv"
    min_freq = 2000
    artists, songs, style_list, style_mapping, df = pre.read_data(file_path, min_freq)
    setups = ["regular", "lemm"]

    for s in setups:
        print(" The Seutp is {}".format(s))
        lemma_path = "data/lemma_songs"
        lemma_songs = pre.lemmatize_texts(songs, lemma_path)

        if s == "lemm":
            df["songs"] = lemma_songs
            print("Regular - {}".format(songs[12]))
            songs = lemma_songs
            print("\n")
            print("A one time example")
            print("lemma - {}".format(songs[12]))
            print("\n")
        print("######## artist identification ########")
        balanced_datasets = pre.prepare_balanced_author_identification_datasets(df, 110)
        train.artist_id(balanced_datasets, model_name)
        embedding_matrix_target_path = "data/embedding_matrix_{}".format(s)
        embedding_matrix = pre.encode_data(songs, model_name, embedding_matrix_target_path)
        print(style_mapping)
        embedding_tsne_target_path = "data/tsne_matrix_{}".format(s)
        embedding_tsne = exp.fit_tsne(embedding_matrix, embedding_tsne_target_path)
        print("T-SNE shape is {}".format(embedding_tsne.shape))
        tsne_figure_target_path = "data/tsne_figure_{}".format(s)
        exp.visualize_tsne_with_clusters(style_list, style_mapping, embedding_tsne, tsne_figure_target_path)
        tsne_pairs_figure_target_dir = "data"
        print(embedding_tsne.shape, " ", len(style_list))
        exp.visualize_pairs(style_list, style_mapping, embedding_tsne, tsne_pairs_figure_target_dir, s)
        print("######## style identification ########")
        train.train_and_evaluate_classifier(embedding_matrix, style_list, style_mapping.values(), s)
        exp.calc_textual_diversity(songs, style_list, style_mapping)
        train.present_tm_results(songs, style_list, style_mapping, num_topics=30, num_words=50, setup=s)
        print("#######################################################################################################")


    