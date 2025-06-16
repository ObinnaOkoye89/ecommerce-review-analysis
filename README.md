# E-Commerce Reviews Analysis with Text Embeddings

Welcome to this project, where we analyze customer reviews from a women's clothing e-commerce dataset using text embeddings. In this project, we:

- **Create and store text embeddings** for the reviews.
- **Perform dimensionality reduction** to visualize the embeddings in 2D.
- **Categorize feedback** based on keywords (e.g. quality, fit, style, comfort).
- **Implement a similarity search** function to retrieve the closest reviews to a given input.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Solution Code](#solution-code)
  - [1. Load and Clean the Dataset](#1-load-and-clean-the-dataset)
  - [2. Create and Store the Embeddings](#2-create-and-store-the-embeddings)
  - [3. Dimensionality Reduction & Visualization](#3-dimensionality-reduction--visualization)
  - [4. Feedback Categorization](#4-feedback-categorization)
  - [5. Similarity Search Function](#5-similarity-search-function)
- [Usage](#usage)
- [License](#license)

---

## Overview

The goal of this project is to leverage the power of text embeddings and Python libraries to extract insights from customer reviews. We use the **OpenAI API** to generate embeddings for each review in the dataset, then reduce the dimensionality for a 2D visualization using **UMAP**. Next, we identify reviews that mention key topics, and finally, we build a function that finds the most similar reviews to a given text input.

---

## Prerequisites

Before getting started, ensure you have Python installed (preferably Python 3.8 or later) along with the following Python libraries:

- `openai==1.3.0`
- `chromadb==0.4.17`
- `pysqlite3-binary==0.5.2`
- `pandas`
- `numpy`
- `umap-learn`
- `matplotlib`
- `scikit-learn`

Also, make sure you have set your OpenAI API key as an environment variable. For example, in your terminal or shell:

```bash
export OPENAI_API_KEY=your_openai_api_key
