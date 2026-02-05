<div align="center">

<img src="https://atrium-research.eu/assets/content/en/pages/communications-kit/clay.png" width="400" alt="Atrium Logo">

# 🌍 Subtask 4.5.2 - Geotagging of texts 🌍

In this repository we include the material for the subtask 4.5.2. You can find more information on our [Technincal Progress report](https://docs.google.com/document/d/1wVVLa_VDjQ80_owQgyJEBFAMbfT7qFW0Co68CIXUag8/edit?tab=t.0#heading=h.atj12slhvkte).

</div>

# 🛠️ Installation
1. **Clone the repository**
   ```cmd
   git clone https://github.com/NikolasKapr/Subtask-4.5.2-Geotagging-of-texts.git
   ```
2. **Navigate to the project directory**
   ```cmd
   cd code/Subtask-4.5.2-Geotagging-of-texts-main
   ```
3. **(Optional but recommended) Create and activate a virtual environment**
   
   Using `virtualenv` (MacOS):
   ```cmd
   python3.11 -m venv venv
   source ./venv/bin/activate
   ```

   Windows:
   ```cmd
   py -3.11 -m venv venv
   venv\Scripts\activate
   ```

5. Upgrade pip
   ```cmd
   pip install --upgrade pip
   ```

6. Install the required dependencies
   ```cmd
   pip install -r requirements.txt
   ```

7. Import your `.xml` file into Recogito Studio.
   
📚 For more information on how to set up a virtual environment, visit the official documentation:
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
  

# 🔧 Workflow
## Step 1: Text Extraction

Using the [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) python library we extract the text from the [Digital Periegisis](https://www.periegesis.org).

**Input**: Extracted text from Digital Periegisis.

**Output**: Text in [JSON Lines (jsonl)](https://jsonlines.org/) format where each line is a separate dictionary with the following structure:

- **`"text"`**: A string containing the paragraph.
- **`"metadata"`**: A dictionary holding the chapter number and the book number of each paragraph.


### Example

```json
{
   "text":"The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "metadata":[
      {
         "chapter":1,
         "book":1
      }
   ]
}
```

## Step 2: Name Entity Recognition (NER)

Take the output of the Text Extraxtion Step as input and perform NER using [NameTag 3](https://lindat.mff.cuni.cz/services/nametag/) service in order to find possible locations.

**Input**: Output from previous step.

**Output**: sentences of cleaned text in [JSON Lines (jsonl)](https://jsonlines.org/) format where each line is a separate dictionary with the following structure:

- **`"text"`**: A string containing the paragraph.
- **`"metadata"`**: A dictionary holding the chapter number and the book number of each paragraph.

- **`"nametag_mentions"`**: A list of possible mentions.
  

### Example

```json
{
   "text":"The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "metadata":[
      {
         "chapter":1,
         "book":1
      }
   ],
   "nametag_mentions":[
      "Peiraeus",
      "Athenians",
      "..."
   ]
}
```
## Step 3: Recontextualisation using LLMs

Take as input the output of Name Entity Recognition (NER) step and using an LLM and a prompt we create a description of each of the mentions. In our case, we used the [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) model with a quantization of Q4_K_M[^1] from [Ollama](https://ollama.com/library/qwen3:14b).

**Input**: Output from previous step.

**Output**: The output is a [JSON Lines (jsonl)](https://jsonlines.org/) file format where each line is a separate dictionary with the following structure:

- **`"text"`**: A string containing the paragraph.
- **`"metadata"`**: A dictionary holding the chapter number and the book number of each paragraph.

- **`"nametag_mentions"`**: A list of possible mentions.

- **`"mentions_tagged"`**: A dictionary holding the mention (from the nametag_mentions), the start character index from the text, the end character index from the text, and the output text from the LLM.

### Example

```json
{
   "text":"The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "metadata":[
      {
         "chapter":1,
         "book":1
      }
   ],
   "nametag_mentions":[
      "Peiraeus",
      "Athenians",
      "..."
   ],
   "mentions_tagged":[
      {
         "name":"Peiraeus",
         "start":4,
         "end":12,
         "recontext":"Peiraeus: Peiraeus was a deme of ancient Attica that became the primary port of Athens..."
      },
       "..."
   ]
}
```
[^1]: Q4_K_M is a widely recommended quantization method for large language models (LLMs) that offers a solid balance between model size, inference speed, and output quality.

## Step 4: Fast Approximate Retrieval

Take as input the output of Recontextualisation using LLMs step and use a [FAISS index](https://github.com/facebookresearch/faiss) that was built from [ToposText](https://topostext.org/) to link the correct mention to the a Knowledge Base.

**Input**: Output from previous step.

**Output**: The output is a [JSON Lines (jsonl)](https://jsonlines.org/) file format where each line is a separate dictionary with the following structure:

Each line contains the following keys:

- **`"text"`**: A string containing the paragraph.
- **`"metadata"`**: A dictionary holding the chapter number and the book number of each paragraph.

- **`"nametag_mentions"`**: A list of possible mentions.
- **`"mentions_tagged"`**: A dictionary holding the mention (from the nametag_mentions), the start character index from the text, the end character index from the text, and the output text from the LLM.
- **`"vector_db"`**: A list of lists that each element holds the ToposText ID and the distance (how close/similar it is to the query, it's something like a score).

### Example

```json
{
   "text":"The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
   "metadata":[
      {
         "chapter":1,
         "book":1
      }
   ],
   "nametag_mentions":[
      "Peiraeus",
      "Athenians",
       "..."
   ],
   "mentions_tagged":[
      {
         "name":"Peiraeus",
         "start":4,
         "end":12,
         "recontext":"Peiraeus: Peiraeus was a deme of ancient Attica that became the primary port of Athens...",
         "vector_db":[
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
         ]
      },
       "..."
   ]
}
```


## (OPTIONAL) Step 4.1: Reranker

Take as input the output of Fast Approximate Retrieval step and use a reranker to further enhance the results. In our case, we used the [Qwen-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) from [🤗 Hugging Face](https://huggingface.co/).

**Input**: Output from previous step.

**Output**: The output is a [JSON Lines (jsonl)](https://jsonlines.org/) file format where each line is a separate dictionary with the following structure:

Each line contains the following keys:

- **`"text"`**: A string containing the paragraph.
- **`"metadata"`**: A dictionary holding the chapter number and the book number of each paragraph.

- **`"nametag_mentions"`**: A list of possible mentions.
- **`"mentions_tagged"`**: A dictionary holding the mention (from the nametag_mentions), the start character index from the text, the end character index from the text, and the output text from the LLM.
- **`"vector_db"`**: A list of lists that each element holds the ToposText ID and the distance (how close/similar it is to the query, it's something like a score).
- **`"reranker"`**: A dictionary holding a list which holds the ToposText ID and the distance.

### Example

```json
{
  "text": "The Peiraeus was a deme from early times, though it was not a port before Themistocles became an archon of the Athenians...",
  "metadata": [
    {
      "chapter": 1,
      "book": 1
    }
  ],
  "nametag_mentions": [
    "Peiraeus",
    "Athenians",
    "..."
  ],
  "mentions_tagged": [
    {
      "name": "Peiraeus",
      "start": 4,
      "end": 12,
      "recontext": "Peiraeus: Peiraeus was a deme of ancient Attica that became the primary port of Athens...",
      "vector_db": [
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
      ],
      "reranker": [
        "379236HPei",
        "0.4406093"
      ]
    },
    "..."
  ]
}
```

## Step 5: Recogito Studio

Take as input the output of Reranker step or the output of the Fast Approximate Retrieval step and prepare it so it can be used as input to [Recogito Studio](https://recogitostudio.org/).

**Input**: Output from previous step.

**Output**: The output is a [Extensible Markup Language (xml)](https://www.w3.org/XML/) file format where each file is a book of Pausanias.
