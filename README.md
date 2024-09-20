# ragnews ![](https://github.com/EthanTu2/ragnews/workflows/tests/badge.svg?dummy=timestamp)

## What is ragnews?

`ragnews` is a Python-powered Question & Answer (Q&A) system built on the Groq API, leveraging Retrieval-Augmented Generation (RAG) to enhance the capabilities of Groq models. It retrieves and processes news articles from a user-provided database and uses them to answer user queries with a focus on delivering accurate and timely responses.

## Instructions

To get started with `ragnews`, follow these steps to set up your
development environment and run the application:

1. **Create a virtual environment:**

```
$ python3.9 -m venv venv
$ source venv/bin/activate
```

2. **Install the necessary Python packages:**

```
$ pip3 install -r requirements.txt
```

3. **Create a GROQ API key:**
    - Create an API key at https://groq.com/
    - Create a `.env` file to include your Groq API key.
    - inside the file, paste `GROQ_API_KEY=(put your API key here)` inside
    - Export the variables:

        ```
        $ export $(cat .env)
        ```

### Example Usage

```
$ python3 ragnews.py 
ragnews> What is the current democratic presidential nominee?
The current Democratic presidential nominee is Kamala Harris.
```