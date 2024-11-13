# readAlike

## Overview

`readAlike` is a book recommendation system that provides similar books to a given input book. The system leverages multiple techniques, including TF-IDF vectorization, Sentence-BERT embeddings, and Approximate Nearest Neighbors (ANN) for generating content-based book recommendations based on title, description, author, and category data.

## Project Structure

### 1. Classes
- **preprocessing/**: Manages data preprocessing.
  - **`Preprocessor`**: Handles data cleaning and formatting from a CSV file of book data.
- **core/**: Contains the core modules for recommendation.
  - **`Library`** and **`Book`**: Models the library of books and individual book data.
  - **`Vectorizer`**: Converts book text data into numerical vectors using TF-IDF and Sentence-BERT.
  - **`DimensionalityReducer`**: Reduces the dimensionality of TF-IDF vectors using Truncated SVD.
  - **`Ann`**: Creates an Approximate Nearest Neighbors model for efficient similarity search.
  - **`Recommender`**: Main recommendation engine that integrates the above components to provide recommendations.

### 2. Key Components
- **`config.py`**: Configuration file with column names for title, description, authors, and categories.
- **`main.py`**: Main entry point for running the recommendation pipeline.

## Program Flow

1. **Preprocessing**: The `Preprocessor` class reads the dataset and performs data cleaning.
2. **Library Initialization**: `Library` is initialized with the cleaned dataset, storing each book as a `Book` object.
3. **Vectorization**: `Vectorizer` creates TF-IDF and Sentence-BERT embeddings for each book.
4. **Dimensionality Reduction**: `DimensionalityReducer` reduces TF-IDF embeddings for optimized ANN performance.
5. **ANN Construction**: `Ann` constructs an ANN model based on the reduced vectors.
6. **Recommendation**: `Recommender` classifies recommendations into TF-IDF, SBERT, and ANN-based results, outputting top similar books.

## Classes and Methods

### `Preprocessor`
- **Attributes**:
  - `df`: DataFrame containing cleaned book data.
- **Methods**:
  - `preprocess_data()`: Cleans and formats the data.
  - `drop_items_with_short_entries()`, `drop_duplicates()`, `convert_strings_into_lists()`: Helper functions to clean the dataset.

### `Library`
- **Attributes**:
  - `books`: List of `Book` objects.
- **Methods**:
  - `get_combined_data()`: Concatenates title, description, authors, and categories into a single string per book.
  - `get_book_idx()`: Retrieves the index of a book within the library.

### `Book`
- **Attributes**:
  - `title`, `description`, `authors`, `categories`: Fields describing the book.
- **Methods**:
  - `get_combined_data()`: Combines title, description, authors, and categories into a single string.

### `Vectorizer`
- **Attributes**:
  - `tfidf_matrix`: Sparse matrix of TF-IDF vectors.
  - `sbert_embeddings`: Sentence-BERT embeddings for each book.
- **Methods**:
  - `tfidf_vectorize()`: Vectorizes a book using TF-IDF.
  - `sbert_vectorize()`: Vectorizes a book or library using SBERT.

### `DimensionalityReducer`
- **Attributes**:
  - `reduced_matrix`: Dimensionality-reduced version of the TF-IDF matrix.
- **Methods**:
  - `reduce()`: Reduces a TF-IDF vector to the lower dimension.

### `Ann`
- **Attributes**:
  - `ann_indices`: ANN model for similarity search.
- **Methods**:
  - `get_nearest_neighbors_by_index()`, `get_nearest_neighbors_by_vector()`: Retrieves nearest neighbors by item index or vector.

### `Recommender`
- **Attributes**:
  - `lib`: Library of books.
  - `vectorizer`: Vectorizer instance.
  - `reducer`: Dimensionality reducer instance.
  - `ann`: ANN instance.
- **Methods**:
  - `recommend()`: Provides top recommendations based on TF-IDF, SBERT, and ANN.

## Example Usage

To run the recommendation engine: 
1. Open the command line and execute ``` pip install requirements.txt ```
2. Execute `main.py`. Given an example book, the program will print the top five recommended books based on three methods: TF-IDF, SBERT, and ANN.