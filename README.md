# Build Your Own Search Engine

Code for the "Build Your Own Search Engine" workshop

Video: https://www.youtube.com/watch?v=nMrGK5QgPVE


What we will do: 

* Use Zoomcamp FAQ documents
    * [DE Zoomcamp](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit)
    * [ML Zoomcamp](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit)
    * [MLOps Zoomcamp](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit)
* Create a search engine for retreiving these documents 
* Later the results can be used for a [Q&A RAG system](https://github.com/alexeygrigorev/llm-rag-workshop) 
* [Reference implementation for text search](https://github.com/alexeygrigorev/minsearch)


## Workshop Outline

1. **Preparing the Environment**
2. **Basics of Text Search**
    - Basics of Information Retrieval
    - Introduction to vector spaces, bag of words, and TF-IDF
3. **Implementing Basic Text Search**
    - TF-IDF scoring with sklearn
    - Keyword filtering using pandas
    - Creating a class for relevance search
4. **Embeddings and Vector Search**
    - Vector embeddings
    - Word2Vec and other approaches for word embeddings
    - LSA (Latent Semantic Analysis) for document embeddings
    - Implementing vector search with LSA
    - BERT embeddings 
5. **Combining Text and Vector Search** (5 minutes)
6. **Practical Implementation Aspects and Tools** (10 minutes)
    - Real-world implementation tools:
        * inverted indexes for text search
        * LSH for vector search (using random projections)
    - Technologies:
        * Lucene/Elasticsearch for text search
        * FAISS and and other vector databases


## 1. Preparing the environment

In the workshop, we'll use Github Codespaces, but you can use any env

We need to install the following libraries:

```bash
pip install requests pandas scikit-learn jupyter
```

Start jupyter:

```bash
jupyter notebook
```

Download the data:

```python
import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

Creating the dataframe:

```python
import pandas as pd

df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])
df.head()
```

## 2. Basics of Text Search

- **Information Retrieval** - The process of obtaining relevant information from large datasets based on user queries.
- **Vector Spaces** - A mathematical representation where text is converted into vectors (points in space) allowing for quantitative comparison.
- **Bag of Words** - A simple text representation model treating each document as a collection of words disregarding grammar and word order but keeping multiplicity.
- **TF-IDF (Term Frequency-Inverse Document Frequency)** - A statistical measure used to evaluate how important a word is to a document in a collection or corpus. It increases with the number of times a word appears in the document but is offset by the frequency of the word in the corpus.


## 3. Implementing Basic Text Search

Let's implement it ourselves.


### Keyword filtering

First, keyword filtering:

```python
df[df.course == 'data-engineering-zoomcamp'].head()
```

### Vectorization

For Count Vectorizer and TF-IDF we will first use a simple example

```python
docs_example = [
    "Course starts on 15th Jan 2024",
    "Prerequisites listed on GitHub",
    "Submit homeworks after start date",
    "Registration not required for participation",
    "Setup Google Cloud and Python before course"
]
```

Let's use a count vectorizer first:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(docs_example)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs
```

This representation is called "bag of words" - here we ignore the order of words, just focus on the words themselves. In many cases this is sufficient and gives pretty good results already.

Now let's replace it with `TfidfVectorizer`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(stop_words='english')
X = cv.fit_transform(docs_example)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs.round(2)
```

### Query-Document Similarity

We represent the query in the same vector space - i.e. using the same vectorizer:


```python
query = "Do I need to know python to sign up for the January course?"

q = cv.transform([query])
q.toarray()
```

We can see the words of the query and the words of some document:

```python
query_dict = dict(zip(names, q.toarray()[0]))
query_dict

doc_dict = dict(zip(names, X.toarray()[1]))
doc_dict
```

The more words in common - the better the matching score. Let's calculate it:

```python
df_qd = pd.DataFrame([query_dict, doc_dict], index=['query', 'doc']).T

(df_qd['query'] * df_qd['doc']).sum()
```

This is a dot-product. So we can use matrix multiplication to compute the score:


```python
X.dot(q.T).toarray()
```

Watch [this linear algebra refresher](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/08-linear-algebra.md) if you're a bit rusty on matrix multiplication (don't worry - it's developer friendly)

Bottom line: it's a very fast and effective method of computing similarities


In practice, we usually use cosine similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(X, q)
```

The TF-IDF vectorizer already outputs a normalized vectors, so the results are identical. We won't go into details of how it works, but you can check "Introduction to Infromation Retrieval" if you want to learn more. 

### Vectorizing all the documents

Let's now do it for all the documents:

```python
fields = ['section', 'question', 'text']
transformers = {}
matrices = {}

for field in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=3)
    X = cv.fit_transform(df[field])

    transformers[field] = cv
    matrices[field] = X

transformers['text'].get_feature_names_out()
matrices['text']
```

### Search

Let's now do search with the text field:

```python
query = "I just signed up. Is it too late to join the course?"

q = transformers['text'].transform([query])
score = cosine_similarity(matrices['text'], q).flatten()
```

Let's do it only for the data engineering course:

```python
mask = (df.course == 'data-engineering-zoomcamp').values
score = score * mask
```

And get the top results:

```python
import numpy as np

idx = np.argsort(-score)[:10]
```

Note: [np.argpartition](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) is a more efficient way of doing the same thing

Get the docs:

```python
df.iloc[idx].text
```

### Search with all the fields & boosting + filtering

We can do it for all the fields. Let's also boost one of the fields - `question` - to give it more importance than to others 

```python
boost = {'question': 3.0}

score = np.zeros(len(df))

for f in fields:
    b = boost.get(f, 1.0)
    q = transformers[f].transform([query])
    s = cosine_similarity(matrices[f], q).flatten()
    score = score + b * s
```

And add filters (in this case, only one):

```python
filters = {
    'course': 'data-engineering-zoomcamp'
}

for field, value in filters.items():
    mask = (df[field] == value).values
    score = score * mask
```

Getting the results:

```python
idx = np.argsort(-score)[:10]
results = df.iloc[idx]
results.to_dict(orient='records')
```

### Putting it all together 

Let's create a class for us to use:

```python
class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
```

Using it:

```python
index = TextSearch(
    text_fields=['section', 'question', 'text']
)
index.fit(documents)

index.search(
    query='I just signed up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)
```

You can fild the implementation here too if you want to use it: https://github.com/alexeygrigorev/minsearch


**Note**: this is a toy example for illustrating how relevance search works. It's not meant to be used in production.

## 4. Embeddings and Vector Search

Problem with text - only exact matches. How about synonyms? 

### What are Embeddings?

- **Conversion to Numbers:** Embeddings transform different words, sentences and documents into dense vectors (arrays with numbers).
- **Capturing Similarity:** They ensure similar items have similar numerical vectors, illustrating their closeness in terms of characteristics.
- **Dimensionality Reduction:** Embeddings reduce complex characteristics into vectors.
- **Use in Machine Learning:** These numerical vectors are used in machine learning models for tasks such as recommendations, text analysis, and pattern recognition.

### SVD

Singular Value Decomposition is the simplest way to turn Bag-of-Words representation into embeddings

This way we still don't preserve the word order (because it wasn't in the Bag-of-Words representation) but we reduce dimensionality and capture synonyms.

We won't go into mathematics, it's sufficient to know that SVD "compresses" our input vectors in such a way that as much as possible of the original information is retained. 

This compression is lossy compression - meaning that we won't be able to restore the 100% of the original vector, but the result is close enough.

Example with images:

<img src="http://habrastorage.org/files/855/a65/c62/855a65c624dc4174b526fb5e03b98555.png" />

Let's use the vectorizer for the "text" field and turn it into embeddings 

```python
from sklearn.decomposition import TruncatedSVD

X = matrices['text']
cv = transformers['text']

svd = TruncatedSVD(n_components=16)
X_emb = svd.fit_transform(X)

X_emb[0]
```

For query:

```python
query = 'I just signed up. Is it too late to join the course?'

Q = cv.transform([query])
Q_emb = svd.transform(Q)
Q_emb[0]
```

Similarity between query and the document:

```python
np.dot(X_emb[0], Q_emb[0])
```

Let's do it for all the documents. It's the same as previously, except we do it on embeddings, not on sparse matrices:

```python
score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
list(df.loc[idx].text)
```

### Non-Negative Matrix Factorization

SVD creates values with negative numbers. It's difficult to interpet them. 

NMF (Non-Negative Matrix Factorization) is a similar concept, except for non-negative input matrices it produces non-negative results.

We can interpret each of the columns (features) of the embeddings as different topic/concepts and to what extent this document is about this concept.

Let's use it for the documents:

```python
from sklearn.decomposition import NMF
nmf = NMF(n_components=16)
X_emb = nmf.fit_transform(X)
X_emb[0]
```

And the query:

```python
Q = cv.transform([query])
Q_emb = nmf.transform(Q)
Q_emb[0]
```

We compute the similarity in the same way as previously:

```python
score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
list(df.loc[idx].text)
```

### BERT 

The problem with the previous two approaches is that they don't take into account the word order. They just treat all the words separately (that's why it's called "Bag-of-Words")

BERT and other transformer models don't have this problem.

Let's create embeddings with BERT. We will use the Hugging Face library for that

```bash
pip install transformers tqdm torch
``` 

Use it:

```python
import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Set the model to evaluation mode if not training
```

We need:

- tokenizer - for turning text into vectors
- model - for compressing the text into embeddings

First, we tokenize the text

```python
texts = [
    "Yes, we will keep all the materials after the course finishes.",
    "You can follow the course at your own pace after it finishes"
]
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

Then we compute the embeddings:

```python
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state
```

Now we need to compress the embeddings:

```python
sentence_embeddings = hidden_states.mean(dim=1)
sentence_embeddings.shape
```

And convert them to a numpy array

```python
X_emb = sentence_embeddings.numpy()
```

Note that if use a GPU, first you need to move your tensors to CPU


```python
sentence_embeddings_cpu = sentence_embeddings.cpu()
```

Let's now compute it for our texts. We'll do it in batches. First, we define a function for batching:

```python
def make_batches(seq, n):
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result
```

And use it:

```python
from tqdm.auto import tqdm
texts = df['text'].tolist()
text_batches = make_batches(texts, 8)

all_embeddings = []

for batch in tqdm(text_batches):
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state
        
        batch_embeddings = hidden_states.mean(dim=1)
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings_np)

final_embeddings = np.vstack(all_embeddings)
```

Let's put it into a function:

```python
def compute_embeddings(texts, batch_size=8):
    text_batches = make_batches(texts, 8)
    
    all_embeddings = []
    
    for batch in tqdm(text_batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    
        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state
            
            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)
    
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings
```

And use it:


```python
X_text = compute_embeddings(df['text'].tolist())
```

