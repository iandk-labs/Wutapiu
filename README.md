# Wutapiu v6.2

## Spectral Linguistic Artificial Intelligence Model with Sense Clouds

Zero Mutation on Read

This repository contains the Wutapiu artificial intelligence model, a natural language processing system based on spectral representations and sense clouds. The model implements the principle of Zero Mutation on Read: the response generation phase never modifies knowledge structures. Only text ingestion and explicit user corrections alter the model parameters.

Version 6 introduces periodic same-sense cloud merging. Version 6.2 includes KeyError protection and uses SenseCloud.merge() for robust cloud fusion operations.

### Philosophical Foundations

- The crystal does not carve itself while it reflects.
- What has not been learned cannot be assumed to exist.
- Structure holds kinship; context holds meaning.
- Form is the face; the root is the voice.

## Affiliation

Developed by **Iandk Labs, Department of Computer Engineering**.

Repository: https://github.com/iandk-labs/Wutapiu

## Description

Wutapiu is an artificial intelligence model for natural language processing that constructs linguistic knowledge from text ingestion. Each word is represented as a complex wave in a 128-bin spectral space. Semantic similarity is computed through spectral resonance between word representations.

The model employs sense clouds for automatic polysemy disambiguation through contextual clustering. Each lexeme can maintain multiple clouds representing distinct senses, which are dynamically created and merged based on spectral similarity thresholds.

## Architecture

The system is organized into five principal modules:

1. **Sparse Structures** (`SparseBinaryMatrix`, `MarkovRelationalMatrix`) — Storage of morphological relations and grammatical category transitions
2. **Language Profiles** (`SpanishProfile`, `EnglishProfile`) — Tokenization, morpheme identification, and root candidate generation
3. **Sense Clouds** (`SenseCloud`, `SpectralSenseClouds`) — Contextual disambiguation with pure read mode and periodic merging
4. **Spectral Core** (`SpectralCore`) — Wave representation, resonance, interference, and phase computation
5. **Integrated Core** (`CheyenneLakota`) — Orchestration of all components, ingestion, generation, and persistence

## Features

- Spectral word representation using complex waves in a 128-dimensional space
- Sense clouds for automatic word-sense disambiguation
- Zero Mutation on Read architecture for knowledge integrity during inference
- Sparse relational matrices for morphological structure encoding
- Markov transition modeling of grammatical categories
- Multilingual support: Spanish and English language profiles
- SQLite persistence for learned knowledge
- Automatic corpus processing from text and PDF files
- Learning through explicit user correction with link reinforcement and weakening
- Periodic same-sense cloud merging every 100 ingestion cycles

## Requirements

- Python 3.8 or higher
- NumPy 1.20 or higher
- PyPDF2 1.26 or higher (optional, for PDF file processing)

## Installation

```bash
git clone https://github.com/iandk-labs/Wutapiu.git
cd Wutapiu
pip install -r requirements.txt
```

## Usage

### Interactive Mode

```bash
# Spanish (default)
python WutapiuV6.2.py es

# English
python WutapiuV6.2.py en
```

### Commands

- `>>correct: ...` — Corrects the previous response (English interface)
- `>>corrige: ...` — Corrects the previous response (Spanish interface)
- `quit`, `exit`, or `salir` — Terminates execution and persists the model state

### Corpus Loading

Place text files (`.txt`) or PDF documents (`.pdf`) in the `corpus/` directory. The system automatically detects and processes new corpus files upon initialization.

### Programmatic Interface

```python
from WutapiuV6_2 import CheyenneLakota

# Initialize the AI model
model = CheyenneLakota("en")

# Ingest textual data
model.ingest("The cat sleeps on the sofa. Dogs bark loudly.")

# Retrieve synonyms
synonyms = model.find_synonyms("cat")
print(synonyms[:5])

# Generate a response
response = model.generate_response("What does the cat do?")
print(response)

# Persist and close
model.close()
```

## Version 6.2 Changes

- KeyError protection in `weaken_link()` method with orphan link removal
- Robust cloud merging through `SenseCloud.merge()` method
- Corrected spectral link persistence mechanisms
- Spectral resonance-based cloud merging instead of cosine similarity

## File Structure

```
Wutapiu/
├── WutapiuV6.2.py      # Main AI model implementation
├── README.md           # This document
├── LICENSE             # License terms (GNU AGPL v3.0)
├── CITATION.cff        # Citation metadata
├── requirements.txt    # Dependency specifications
├── .gitignore          # Git ignore patterns
├── corpus/             # Corpus directory (auto-created)
└── wutapiu_<lang>.db   # SQLite knowledge base (auto-created)
```

## Theoretical Framework

### Spectral Resonance

A similarity measure between two complex waves, computed as the normalized complex inner product. Values range from 0.0 (no resonance) to 1.0 (maximum resonance).

### Interference

Indicates phase alignment between spectral representations. Values range from -1.0 (destructive interference) to +1.0 (constructive interference).

### Spectral Phase

Represents the semantic orientation of a word within the spectral space, computed from the weighted angular distribution of the spectrum.

### Vocabulary Identity

The spectrum of a word minus the vocabulary mean spectrum, emphasizing the distinctive features of each lexical item.

### Sense Cloud

A contextual cluster associated with a specific lexeme. Each cloud accumulates evidence from contexts where the lexeme appears, representing a distinct sense. Cloud mass indicates the accumulated evidence weight.

## License

This work is distributed under the GNU Affero General Public License v3.0. See the LICENSE file for complete terms.

## Citation

When using this model in research or academic work, please cite as follows:

```bibtex
@software{wutapiu_v6_2,
  author       = {Iandk Labs},
  title        = {Wutapiu v6.2: Spectral Linguistic Artificial Intelligence Model with Sense Clouds},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  institution  = {Iandk Labs, Department of Computer Engineering},
  version      = {v6.2},
  doi          = {[DOI_TO_BE_ASSIGNED]},
  url          = {https://github.com/iandk-labs/Wutapiu}
}
```

APA format:

> Iandk Labs. (2026). *Wutapiu v6.2: Spectral Linguistic Artificial Intelligence Model with Sense Clouds* (v6.2) [AI model]. Iandk Labs, Department of Computer Engineering. https://github.com/iandk-labs/Wutapiu

## Limitations

- The model learns exclusively from ingested text; no pre-trained knowledge is included
- Sufficient text volume is required for robust sense cloud formation
- Response generation is associative rather than propositional, producing chains of spectrally related terms
