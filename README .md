<h1 align="center">Potential Talents · HR Candidate Finder</h1>

<p align="center">
An NLP portfolio project comparing seven text representation methods to rank HR-related profiles and refine candidate discovery through recruiter feedback.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-NLP-blue" alt="Python NLP">
  <img src="https://img.shields.io/badge/Methods-7embedding methods-purple" alt="Seven embedding methods">
  <img src="https://img.shields.io/badge/Ranking-Cosine%20Similarity-orange" alt="Cosine similarity">
  <img src="https://img.shields.io/badge/Interface-Gradio-green" alt="Gradio interface">
  <img src="https://img.shields.io/badge/Feedback-Star%20Re--ranking-yellow" alt="Star re-ranking">
</p>

> **Project at a glance:** Prepared 52 distinct job-title profiles, compared semantic ranking approaches, explored relevance cut-offs, and demonstrated a Gradio interface that updates candidate rankings when users star preferred profiles. An additional local RAG experiment explored candidate justifications.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Questions](#business-questions)
- [Project Workflow](#project-workflow)
- [1. Data Preparation](#1-data-preparation)
- [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
- [3. Text Preprocessing](#3-text-preprocessing)
- [4. Vectorization and Ranking](#4-vectorization-and-ranking)
- [5. Method Comparison and Results](#5-method-comparison-and-results)
- [6. Filtering and Cut-off Selection](#6-filtering-and-cut-off-selection)
- [7. Star-based Re-ranking](#7-star-based-re-ranking)
- [8. Gradio Application](#8-gradio-application)
- [9. Local RAG Experiment](#9-local-rag-experiment)
- [Bias, Limitations, and Next Steps](#bias-limitations-and-next-steps)
- [Project Contribution](#project-contribution)
- [Conclusion](#conclusion)

---

## Project Overview

Recruiters often search for candidates using short phrases, while profiles describe experience, interests, and career goals in different ways. Exact keyword matching can miss related language; semantic similarity can retrieve broader matches but also introduce irrelevant results.

This project investigates that trade-off using two queries:

- **“aspiring human resources”**
- **“seeking human resources”**

The objective is to support candidate discovery: rank profiles by relevance, let the recruiter inspect the shortlist, and use starred profiles to refine the results. Similarity is a measure of textual relevance—not evidence that a person is qualified or suitable for hiring.

The implementation uses pretrained models for inference rather than training a neural network from scratch. It combines data preparation, exploratory analysis, comparative NLP experiments, and a user-facing prototype.

## Business Questions

| Question | Project response |
|---|---|
| How robust is the ranking approach? | Compare seven methods across two queries, inspect returned profiles, and assess the limits of a small, manually labelled sample. |
| How does starring change the shortlist? | Blend the original search signal with similarity to selected profiles and compare the initial and adjusted results. |
| How can irrelevant profiles be filtered? | Explore similarity thresholds alongside ranked-score and distribution plots. |
| Can one cut-off generalize to other roles? | Treat thresholds as model- and query-specific; broader validation is needed before reuse. |
| How can human bias be reduced? | Examine representation in the data, recognize location leakage, and propose job-related inputs and ranking audits. |

## Project Workflow

| Stage | Main activity | Output |
|---|---|---|
| 1. Prepare | Inspect records, remove duplicate titles, clean fields | 52 retained profiles |
| 2. Explore | Examine title length, location, and connections | Distribution plots |
| 3. Process text | Combine title and location, tokenize, lemmatize, lowercase | Clean text and manual relevance labels |
| 4. Compare representations | BoW, TF-IDF, GloVe, BERT, SBERT, E5, BGE | Candidate vectors and similarity scores |
| 5. Review rankings | Compare top results for two search queries | Method comparison and relevance checks |
| 6. Explore filtering | Inspect score distributions and ranked curves | Candidate cut-off experiments |
| 7. Refine with feedback | Select preferred profiles in the interface | Adjusted top-10 shortlist |
| 8. Extend | Test retrieval with a small local language model | Experimental candidate explanations |

The notebooks document the preparation and ranking experiments. The screenshots demonstrate the separate app prototype; its source is not included in the notebook set documented here.

## 1. Data Preparation

Preparation included:

- Dropping the unused fit column.
- Normalizing the recorded 500+ connection value to 500.
- Removing duplicate records using job_title as the deduplication key.
- Identifying non-ASCII location characters and replacing selected Turkish characters.
- Retaining profile IDs to support traceability.
- Adding 52 manually assigned binary relevance labels for later assessment.

**Important interpretation:** the 52 records represent distinct retained job titles. Deduplicating by title can merge different people who share a title, so this is not a verified count of unique individuals.

## 2. Exploratory Data Analysis

EDA examined three aspects of the dataset:

| Analysis | Purpose |
|---|---|
| Job-title character length | Understand how much text each profile provides. |
| Location distribution | Identify geographic concentration and representation differences. |
| Connection distribution | Inspect the spread of connection counts and the capped category. |

These plots provide context for interpreting the rankings. They do not, on their own, establish whether rankings are biased; that requires comparing model outputs across groups.

## 3. Text Preprocessing

The ranking text combines job title and location. Despite the column name clean_job_title, the cleaned input therefore contains both fields. Connection count is retained separately and is not concatenated into the ranking text in this implementation.

A spaCy pipeline:

1. Tokenizes each profile's combined text.
2. Removes punctuation and stopwords.
3. Lemmatizes the remaining tokens.
4. Converts them to lowercase.
5. Joins them into a string for each vectorizer or model's own tokenizer.

This creates a consistent candidate representation across methods, but it also introduces a trade-off: aggressive cleaning may remove useful context for transformer models. Candidate text is cleaned while query strings are passed directly to the ranking functions, so words such as “seeking” may appear as “seek” in profiles but remain “seeking” in queries. A future comparison should test consistent light cleaning against the existing pipeline.

## 4. Vectorization and Ranking

Seven approaches convert text into numerical representations:

| Method | Implementation | What it contributes |
|---|---|---|
| Bag of Words | CountVectorizer | An interpretable baseline based on word counts and overlap. |
| TF-IDF | TfidfVectorizer | Weights words by their distinctiveness in the candidate corpus. |
| GloVe | glove-wiki-gigaword-100 ; average available word vectors | Dense semantic representations from pretrained word embeddings. |
| BERT | bert-base-uncased; mean token pooling | Contextual token representations used as a ranking baseline. |
| SBERT | sentence-transformers/all-MiniLM-L6-v2; masked mean pooling | Compact embeddings designed for sentence-level similarity. |
| E5 | intfloat/e5-small-v2; masked mean pooling | A small pretrained text embedding model for semantic retrieval experiments. |
| BGE | BAAI/bge-small-en-v1.5; masked mean pooling | An additional compact semantic embedding comparison. |

For every method, cosine similarity compares each candidate vector with the query vector. Candidates are then sorted by descending similarity.

Higher similarity means closer text representations within that method. It is not a probability of hiring success, and score magnitudes cannot be used to decide which model is best across different embedding spaces.

## 5. Method Comparison and Results

### What the saved rankings show

| Method | First result for “aspiring human resources” | First result for “seeking human resources” |
|---|---|---|
| BoW | Aspiring HR Manager, seeking internships | Aspiring HR Manager, seeking internships |
| TF-IDF | Aspiring HR Manager, seeking internships | Aspiring HR Manager, seeking internships |
| GloVe | Aspiring HR Manager, seeking internships | Aspiring HR Manager, seeking internships |
| BERT | Seeking HR Opportunities | Seeking HR Opportunities |
| SBERT | Aspiring HR Professional | Seeking HR Position |
| E5 | Aspiring HR Specialist | Seeking HR Opportunities |
| BGE | Aspiring HR Professional | Seeking HR Opportunities |

Titles are shortened here for readability. The notebook outputs contain the corresponding cleaned profiles and scores.

### Metrics checks from saved outputs 

Seven text representation methods were evaluated using Precision@5, Precision@10, Mean Reciprocal Rank (MRR), Average Precision (AP), and NDCG@10.

The following figures summarise the recorded ranking evaluation for each method.

| Method | P@5 | P@10 | MRR | AP | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| BoW | 1.00 | 1.00 | 1.00 | 0.977855 | 1.000000 |
| TF-IDF | 1.00 | 1.00 | 1.00 | 0.977855 | 1.000000 |
| GloVe | 1.00 | 1.00 | 1.00 | 0.955214 | 1.000000 |
| BERT | 0.80 | 0.70 | 1.00 | 0.831193 | 0.753449 |
| SBERT | 1.00 | 1.00 | 1.00 | **0.995338** | 1.000000 |
| E5 | 1.00 | 1.00 | 1.00 | 0.990001 | 1.000000 |
| BGE | 1.00 | 1.00 | 1.00 | 0.987924 | 1.000000 |

# Understanding the Metrics

- Precision@5 / Precision@10: Proportion of relevant candidates within the first five or ten results.
- MRR: Measures how early the first relevant candidate appears.
- AP: Measures how consistently relevant candidates appear near the top across the full ranking.
- NDCG@10: Evaluates the ordering of relevant candidates within the top ten results.

# Key Findings
SBERT achieved the highest AP (0.995338), followed by E5 and BGE, indicating the strongest overall ordering of relevant candidates in this evaluation.

All methods except BERT achieved perfect Precision@5, Precision@10, and NDCG@10. BERT returned four relevant candidates in the top five and seven in the top ten. Every method placed a relevant candidate first, resulting in an MRR of 1.00.

The strong performance of BoW and TF-IDF shows that simple lexical methods remain competitive for this dataset. These results reflect the evaluated candidate set and relevance labels; testing on more queries and a larger dataset would help assess generalisation.

## 6. Filtering and Cut-off Selection

The comparison notebook explores a similarity threshold of 0.3, while the E5 analysis also tests 0.4. Histograms show the score distribution, and sorted-score curves help inspect where relevance may begin to decline. The TF-IDF outputs illustrate the effect of filtering: the same 0.3 threshold returns eight profiles for the aspiring query and only two for the seeking query.

A cut-off is therefore a retrieval setting, not a universal relevance rule. It should be selected using labelled examples for each model and query family. Percentile-based shortlists and top-k retrieval are useful alternatives to test, but neither guarantees relevance. Generalization to other roles remains unvalidated.

## 7. Star-based Re-ranking

The interface lets users select preferred profiles from the initial top 10. The guide's feedback mechanism combines the original query with a positive anchor formed from the selected profiles.

The screenshots demonstrate adjusted results after two profiles are selected. The formula describes the intended mechanism; the app source would be needed to verify its exact implementation.

### Before and after feedback

In the captured “Human resources” search, the user selects an HR staffing/recruiting profile and an aspiring HR professional profile.

| Profile | Initial rank | Adjusted rank |
|---|---:|---:|
| HR Staffing and Recruiting Professional | 1 | 1 |
| Aspiring HR Professional — inclusive work environments | 2 | 2 |
| Aspiring HR Professional — team-focused | 8 | 4 |
| HR Professional — Greater Boston | 10 | 5 |
| Aspiring HR Manager — graduating 2020 | 4 | 10 |

The shortlist changes in response to the user's preferences. This demonstrates personalization, not a measured improvement in hiring outcomes. Starring can also reinforce a user's existing preferences, so the adjusted ranking still needs review.

## 8. Gradio Application

The HR Candidate Finder prototype presents the workflow through:

- A text search field and Search button.
- An initial top-10 results table with similarity scores.
- Star-style checkboxes for selecting preferred profiles.
- An adjusted top-10 table after feedback.

## Project Article

Read my [project walkthrough on Medium](https://medium.com/@safwtsafwt19/building-an-hr-candidate-finder-from-keyword-matching-to-semantic-search-f09b1b86a26d?sharedUserId=safwtsafwt19) for a step-by-step look at the vectorization methods, model comparison, starring mechanism, and project limitations.

## 9. Local RAG Experiment

An optional experiment explores whether a small local language model can explain retrieved profiles. The prompt asks the model to answer using retrieved context and acknowledge uncertainty. The saved output nevertheless includes unsupported interpretations and exceeds the requested two-sentence limit. 

This experiment highlights a practical boundary: retrieval and generation do not automatically produce reliable explanations. Short chunks can separate profile context, and a small model may infer facts absent from the source. The RAG chain also uses a separate retriever; it is not shown consuming the main ranker's exact top-three candidates.

## Bias, Limitations, and Next Steps

| Limitation | Implication | Next step |
|---|---|---|
| Small sample and two queries | Limited evidence of robustness beyond HR searches. | Evaluate more roles, query variants, and independent profiles. |
| Broad manual binary labels | Strong HR relevance can conceal weak seniority or intent matching. | Use query-specific, graded labels and independent review. |
| Location included in embeddings | Geography can influence similarity even without an explicit location filter. | Compare title-only inputs with title-plus-location rankings. |
| Connection distribution inspected only | EDA does not establish whether rankings correlate with network size. | Audit score and rank relationships with connection count. |
| Title-based deduplication | Different people sharing a title may be removed. | Deduplicate using stable person IDs and verified record rules. |
| Nonstandard model input and pooling choices | Current results are implementation-specific baselines. | Verify each model's prescribed prefixes, pooling, and inference settings, then rerun comparisons. |
| Aggressive candidate cleaning | Context and query alignment may be lost. | Compare light cleaning with the current spaCy pipeline. |
| Feedback reflects user preferences | Stars may amplify bias rather than reduce it. | Keep feedback transparent and audit changes in shortlist composition. |
| Missing app source and full metrics | Screenshots demonstrate behavior but do not provide reproducible deployment. | Add app code, pinned dependencies, complete evaluation, and a tested launch command. |

## Project Contribution

My contribution covered the analytical workflow and its translation into a usable candidate-discovery prototype:

- **Problem framing:** translated a recruitment search task into query-to-profile ranking.
- **Data preparation:** cleaned profile fields, handled repeated titles, and created traceable processed datasets.
- **Exploratory analysis:** examined text length, geographic representation, and connection distributions.
- **NLP implementation:** compared seven approaches from sparse word counts to pretrained contextual embeddings.
- **Evaluation:** inspected ranked profiles for two query intentions and created manual relevance labels.
- **Decision support:** explored filtering thresholds and score distributions rather than treating similarity as a hiring decision.
- **Interface design:** demonstrated search, profile selection, and adjusted rankings in a Gradio prototype.
- **Experimentation:** built a separate local RAG workflow and identified limits in generated candidate explanations.

The project demonstrates practical work with **Python, pandas, spaCy, scikit-learn, Gensim, PyTorch, Hugging Face Transformers, Gradio, LangChain, and Chroma**, alongside the judgment needed to interpret results and communicate their limits.

Although this project uses an HR dataset, the underlying approach can be adapted across domains. In e-commerce, it could rank products against a customer’s search and refine results based on selected products. In business, it could match supplier profiles to purchasing requirements. By adapting the data and evaluating relevance for each use case, the same workflow—text representation, similarity ranking, and user feedback—can support a broader range of search and matching applications.

## Conclusion

Potential Talents turns a recruitment search question into a structured NLP workflow: prepare profiles, compare representations, inspect rankings, explore cut-offs, and refine a shortlist through user feedback.

The saved experiments show that multiple approaches retrieve HR-related profiles effectively under the project's manual labels, while their ordering and sensitivity to search intent differ. The app screenshots make the feedback mechanism visible, and the RAG experiment adds a useful lesson about the difference between retrieving relevant text and generating trustworthy explanations.

For hiring managers and recruiters, this project demonstrates my ability to connect **business needs, data preparation, NLP implementation, evaluation, and interface design**. Its value lies in both building a working analytical prototype and explaining what the evidence supports, what remains uncertain, and how the solution should be strengthened before wider use.
