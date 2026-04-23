<div align="center">

<img src="https://atrium-research.eu/assets/content/en/pages/communications-kit/clay.png" width="400" alt="Atrium Logo">

# 🌍 Subtask 4.5.2 - Geotagging of texts 🌍

In this repository we include the material for the subtask 4.5.2.

You can find more information on our [Technincal Progress report](https://docs.google.com/document/d/1wVVLa_VDjQ80_owQgyJEBFAMbfT7qFW0Co68CIXUag8/edit?tab=t.0#heading=h.atj12slhvkte).

</div>

# 🛠️ Installation
1. **Clone the repository**
   ```cmd
   git clone https://github.com/atrium-research/T4.5.2_Geotagging_of_texts.git
   ```
2. **Navigate to the project directory**
   ```cmd
   cd notebooks
   ```
3. **(Optional but recommended) Create and activate a virtual environment**
   
   Using `virtualenv` (MacOS and Linux):
   ```cmd
   python3.11 -m venv venv
   source ./venv/bin/activate
   ```

   Windows:
   ```cmd
   py -3.11 -m venv venv
   venv\Scripts\activate
   ```
4. **Install an LLM inference tool ([llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master))**
   
   MacOS and Linux:
   ```cmd
   brew install llama.cpp
   ```
   
   Windows:
   ```cmd
   winget install llama.cpp
   ```
   
📚 For more information on how to set up a virtual environment, visit the official documentation:
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
  

# 🔧 Workflow
## Step 1: Name Entity Recognition (NER)

A deep learning model [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) using [spaCy](https://spacy.io/) framework was trained for NER due to its strong performance in delivering accurate results and its efficient resource requirements.

**Input**: Text chunks in [JSON Lines (jsonl)](https://jsonlines.org/) file format.

**Output**: Sentences of cleaned text in [JSON Lines (jsonl)](https://jsonlines.org/) format where each line is a separate dictionary with the following structure:

- **`"text"`**: A string containing the paragraph.
- **`"spans"`**: A list of possible mentions.
  

### Example

```json
{
   "text": "The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "mentions":[
      {"start":4, "end":12, "mention":"Peiraeus", "label":"LOCATION"},
      "..."
   ]
}

```
## Step 2: Text Generation

Take as input the output of Name Entity Recognition (NER) step and using an LLM and a prompt we create a description of each of the mentions. In our case, we used the [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b) model.

**Input**: Output from previous step.

**Output**: The output is a [JSON Lines (jsonl)](https://jsonlines.org/) file format where each line is a separate dictionary with the following structure:

- **`"text"`**: A string containing the paragraph.
- **`"spans"`**: A list of possible mentions.
- **`"recontext"`**: A dictionary where the value is the output of the LLM.

### Example

```json
{
   "text": "The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "spans":[
      {"start":4, "end":12, "mention":"Peiraeus", "label":"LOCATION", "recontext":"Peiraeus: Peiraeus was a deme of ancient Attica that became the primary port of Athens..."},
      "..."
   ]
}

```

## Step 3: Indexing & Fast Approximate Retrieval

Take as input the output of Text Generation step and use a [FAISS index](https://github.com/facebookresearch/faiss) that was built from [ToposText](https://topostext.org/) to link the correct mention to the a Knowledge Base.

**Input**: Output from previous step.

**Output**: The output is a [JSON Lines (jsonl)](https://jsonlines.org/) file format where each line is a separate dictionary with the following structure:

Each line contains the following keys:

- **`"text"`**: A string containing the paragraph.
- **`"spans"`**: A list of possible mentions.
- **`"recontext"`**: A dictionary where the value is the output of the LLM.
- **`"index_results"`**: A list of lists that each element holds the ToposText ID and the distance (how close/similar it is to the query).

### Example

```json
{
   "text": "The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "spans":[
      {"start":4, "end":12, "mention":"Peiraeus", "label":"LOCATION", "recontext":"Peiraeus: Peiraeus was a deme of ancient Attica that became the primary port of Athens...", "index_results":[
            [
               "379236HPei",
               "0.4406093"
            ],
            [
               "379236UKan",
               "0.61706793"
            ],
            [
               "379237HZea",
               "0.6831315"
            ],
             "..."
         ]},
      "..."
   ]
}

```


## Step 4: Reranker

Take as input the output of Fast Approximate Retrieval step and use an LLM as a reranking tool to further enhance the results. In our case, we used the [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b) model.

**Input**: Output from previous step.

**Output**: The output is a [JSON Lines (jsonl)](https://jsonlines.org/) file format where each line is a separate dictionary with the following structure:

Each line contains the following keys:

- **`"text"`**: A string containing the paragraph.
- **`"spans"`**: A list of possible mentions.
- **`"recontext"`**: A dictionary where the value is the output of the LLM.
- **`"index_results"`**: A list of lists that each element holds the ToposText ID and the distance (how close/similar it is to the query).
- **`"reranker_results"`**: A dictionary where the value is the ToposText ID the LLM selected.

### Example

```json
{
   "text": "The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "spans":[
      {"start":4, "end":12, "mention":"Peiraeus", "label":"LOCATION", "recontext":"Peiraeus: Peiraeus was a deme of ancient Attica that became the primary port of Athens...", "index_results":[
            [
               "379236HPei",
               "0.4406093"
            ],
            [
               "379236UKan",
               "0.61706793"
            ],
            [
               "379237HZea",
               "0.6831315"
            ],
             "..."
         ],"reranker_results":
        "379236HPei",
       },
      "..."
   ]
}

```

## Step 5: Recogito Studio

Take as input the output of Reranker step and prepare it so it can be used as input to your target annotation enviroment. In our case, we selected [Recogito Studio](https://recogitostudio.org/).

**Input**: Output from previous step.

**Output**: The output is an [XML/TEI](https://tei-c.org/) file format. However, you can adjust the output format based on the supported formats of your annotation enviroment.
