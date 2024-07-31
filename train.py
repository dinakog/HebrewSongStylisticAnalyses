from collections import defaultdict

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim import corpora
from gensim.models import LdaModel
from lexical_diversity import lex_div as ld
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import pre  

# Stop word
stop_words = [
    "שכן", "אותה", "קרוב", "ו", "האם", "מכיוון", "מבין", "עת", "תחת",
    "נכון", "לאור", "עם", "נגד", "רוב", "במקום", "במשך", "אשר", "כל", "הרבה",
    "מתחת", "מדי", "בזכות", "סביב", "אחרי", "לעומת", "כ", "לה", "הן", "כמה",
    "גם", "ללא", "אצל", "כלל", "מאת", "מרבית", "אף", "יותר", "לו", "לא",
    "מן", "כיוון", "יש", "בקרב", "למען", "את", "למשל", "עצמם", "תוך", "לשם",
    "כי", "מתוך", "בשל", "the", "מעבר", "עוד", "מטעם", "פי", "אלא", "עצמה",
    "מ", "משום", "פחות", "אחר", "לבין", "אלה", "לצד", "זו", "יחד", "עבור",
    "בעוד", "לפני", "הללו", "מה", "הוא", "כן", "לפי", "כדי", "היא", "כנגד",
    "שאר", "עבר", "אבל", "ה", "בתוך", "אי", "כגון", "עקב", "בפני", "בעקבות",
    "בתור", "זה", "בגלל", "מי", "קודם", "of", "מעל", "של", "אך", "לאחר",
    "מנת", "א", "בין", "כמו", "אולם", "ב", "בניגוד", "לגבי", "החל", "כלפי",
    "מספר", "The", "בה", "דרך", "ל", "מאז", "או", "מפני", "על", "לקראת",
    "אותם", "לאורך", "הרי", "אני", "הם", "אלו", "כך", "מעט", "רק", "מהם",
    "למרות", "אותו", "מול", "מאחר", "אם", "מצד", "ליד", "עצמו", "ידי", "זהו",
    "לידי", "בידי", "זאת", "באמצעות", "ככל", "עד", "כאשר", "אל", "מאשר", "כפי",
    "כלומר", "לי", "לך",  "אתה", "אותך", "אז", "אותי"

]


def train_and_evaluate_classifier(embeddings, labels, classes, setup):
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Split the data into train (80%), validation (10%), and test (10%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

    # Train logistic regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print('Validation Classification Report:')
    print(classification_report(y_val, y_val_pred))

    # Evaluate the model on the test set
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print('Test Classification Report:')
    print(classification_report(y_test, y_test_pred))
    save_path = "data/cm_{}.png".format(setup)

    plot_confusion_matrix(y_test, y_test_pred, classes, save_path=save_path)

    return clf, val_accuracy, test_accuracy



def train_and_evaluate_classifier_cv(embeddings, labels, classes, n_splits=5):
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Perform cross-validation with a logistic regression classifier
    clf = LogisticRegression(max_iter=1000)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, embeddings, labels, cv=cv,
                                scoring=['accuracy', 'f1_weighted'],
                                return_estimator=True, return_train_score=True)

    # Print results
    print(f'Cross-Validation Results:')
    print(f'Mean Validation Accuracy: {cv_results["test_accuracy"].mean():.4f} (+/- {cv_results["test_accuracy"].std() * 2:.4f})')
    print(f'Mean Validation F1-score: {cv_results["test_f1_weighted"].mean():.4f} (+/- {cv_results["test_f1_weighted"].std() * 2:.4f})')


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot and save normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def preprocess_text(text):
    tokens = ld.flemmatize(text)
    return [token for token in tokens if token.isalnum() and token not in stop_words]


def perform_topic_modeling_by_style(texts, styles, num_topics=3, num_words=10):
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

    # Get topics for each document
    document_topics = []
    for doc in corpus:
        topics = lda_model.get_document_topics(doc)
        document_topics.append(sorted(topics, key=lambda x: x[1], reverse=True))

    # Get overall topics
    overall_topics = []
    for topic_id in range(num_topics):
        top_words = lda_model.show_topic(topic_id, topn=num_words)
        overall_topics.append((topic_id, [word for word, _ in top_words]))

    # Group topics by style
    style_topics = defaultdict(list)
    for doc_topics, style in zip(document_topics, styles):
        style_topics[style].append(doc_topics)

    # Aggregate topic probabilities for each style
    style_topic_probabilities = {}
    for style, topics_list in style_topics.items():
        style_probabilities = [0] * num_topics
        for doc_topics in topics_list:
            for topic_id, prob in doc_topics:
                style_probabilities[topic_id] += prob
        total_prob = sum(style_probabilities)
        style_topic_probabilities[style] = [prob / total_prob for prob in style_probabilities]
        

    return lda_model, document_topics, overall_topics, style_topic_probabilities


def present_tm_results(songs, style_list, style_map, num_topics=3, num_words=15, save_dir='topic_charts', setup='regular'):
    lda_model, document_topics, overall_topics, style_topic_probabilities = perform_topic_modeling_by_style(songs, style_list, num_topics, num_words)

    print("\nOverall Topics:")
    for topic_id, words in overall_topics:
        print(f"Topic {topic_id}: {', '.join(words)}")

    print("\nStyle Topic Probabilities:")
    for style, probabilities in style_topic_probabilities.items():
        print(f"{style_map[style]}:")
        for topic_id, prob in enumerate(probabilities):
            print(f"  Topic {topic_id}: {prob:.4f}")
            print(f"    Top words: {', '.join(overall_topics[topic_id][1])}")


    # Create and save pie charta for top topic probabilities across styles
    os.makedirs(save_dir, exist_ok=True)

    for style, probabilities in style_topic_probabilities.items():
        plt.figure(figsize=(10, 8))
        plt.pie(probabilities,
                labels=[f'Topic {i}' for i in range(len(probabilities))],
                autopct='%1.1f%%', startangle=90)
        plt.title(f"Topic Distribution for {style_map[style]}")
        plt.axis('equal')

        save_path = os.path.join(save_dir, f"{style_map[style]}_topic_distribution_{setup}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Pie chart for {style_map[style]} saved as {save_path}")
            
            
def artist_id(balanced_datasets, model_name):
    for style in balanced_datasets.keys():
        print(f"\nPreparing and evaluating data for {style}")
        embeddings, labels, classes = pre.prepare_data_for_classification(balanced_datasets, style, model_name)
        train_and_evaluate_classifier_cv(embeddings, labels, classes)





