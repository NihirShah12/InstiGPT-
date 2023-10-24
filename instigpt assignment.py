import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import sys
import string
from collections import defaultdict

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():


    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python final.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding="utf8") as file:
                files[filename] = file.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = nltk.word_tokenize(document.lower())
    words = [word for word in words if word not in string.punctuation and word not in stop_words]
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = defaultdict(int)
    total_documents = len(documents)

    for document in documents.values():
        words_set = set(document)
        for word in words_set:
            idfs[word] += 1

    for word, freq in idfs.items():
        idfs[word] = 1 + (total_documents / (1 + freq))

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf_scores = {}

    for filename, words in files.items():
        tf_idf_scores[filename] = sum(tf(word, words) * idfs[word] for word in query)

    top_matches = sorted(tf_idf_scores.keys(), key=lambda x: tf_idf_scores[x], reverse=True)[:n]
    return top_matches


def tf(word, document):
    """
    Given a word and a list of words, return the term frequency (TF) of that word
    in the document.
    """
    return document.count(word)


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = []

    for sentence, words in sentences.items():
        idf_score = sum(idfs[word] for word in query if word in words)
        query_density = sum(1 for word in query if word in words) / len(words)
        sentence_scores.append((sentence, idf_score, query_density))

    top_matches = sorted(sentence_scores, key=lambda x: (x[1], x[2]), reverse=True)[:n]
    return [match[0] for match in top_matches]


if __name__ == "__main__":
    main()
