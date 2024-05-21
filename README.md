# Build Your Own Search Engine

Code for the "Build Your Own Search Engine" workshop

Register here: https://lu.ma/jsyob4df


What we will do: 

* Use Zoomcamp FAQ documents
    * [DE Zoomcamp](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit)
    * [ML Zoomcamp](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit)
    * [MLOps Zoomcamp](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit)
* Create a search engine for retreiving these documents 
* Later the results can be used for a [Q&A RAG system](https://github.com/alexeygrigorev/llm-rag-workshop) 
* [Reference implementation for text search](https://github.com/alexeygrigorev/minsearch)


## Workshop Outline

1. **Preparing the Environment** (5 minutes)


2. **Basics of Text Search** (10 minutes)

- Basics of Information Retrieval
- Introduction to vector spaces, bag of words, and TF-IDF


3. **Implementing Basic Text Search** (15 minutes)

- TF-IDF scoring with sklearn
- Keyword filtering using pandas

4. **Vector Search** (20 minutes)

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
pip install requests jupyter pandas scikit-learn transformers
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



Let's implement it ourselves.

First, keyword filtering:

```python
df[df.course == 'data-engineering-zoomcamp'].head()
```


For Count Vectorizer and TF-IDF we will first use a simple example

```python
documents = [
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

Now replace it with TfidfVectorizer:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(stop_words='english')
X = cv.fit_transform(docs_example)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs.round(2)
```

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
cosine_similarity(X, q)
```

The TF-IDF vectorizer already outputs a normalized vectors, so the results are identical. We won't go into details of how it works, but you can check "Introduction to Infromation Retrieval" if you want to learn more. 



