# 🔐 PHI De-identification Gateway

**A privacy-preserving clinical text processing gateway using DistilBERT, Microsoft Presidio, regular expressions and local identifier restoration.**

## Overview

The PHI De-identification Gateway identifies and masks Protected Health Information (PHI) in clinical text before it is processed by an external Large Language Model (LLM).

The application maintains a local mapping between original identifiers and replacement placeholders. Only the de-identified clinical text is sent to the external LLM. The mapping remains local and is used to restore identifiers in the generated response.

### Key Features

- Hybrid PHI detection using DistilBERT, Presidio and custom regular expressions.
- Consistent placeholders for repeated identifiers within a note.
- Local identifier-to-placeholder mapping.
- LLM integration using Google Gemini.
- Local response rehydration.
- Presidio baseline comparison.
- Synthetic training and manually labelled synthetic test data.
- Streamlit interface for interactive demonstrations.

## Architecture

```text
                  Original Clinical Note
                           |
                           v
                 PHI Detection Gateway
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
          DistilBERT    Presidio     Regex Rules
             |             |             |
             +-------------+-------------+
                           |
                           v
                  Entity Resolution
                           |
                           v
                   PHI De-identification
                           |
              +------------+------------+
              |                         |
              v                         v
        Masked Clinical Note      Local Mapping
              |
              v
          External LLM
           (Gemini)
              |
              v
         Masked Response
              |
              v
        Local Rehydration
              |
              v
       Restored LLM Response
```

The placeholder mapping is not included in the external LLM request.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| NER model | DistilBERT |
| Base checkpoint | distilbert/distilbert-base-uncased |
| Rule-based detection | Microsoft Presidio |
| Pattern detection | Python regular expressions |
| NLP framework | Hugging Face Transformers |
| Deep learning | PyTorch |
| LLM | Google Gemini API |
| User interface | Streamlit |
| Testing | Pytest |
| Synthetic data | Faker and custom templates |

## Model Details

The project fine-tunes DistilBERT for token-level PHI identification using BIO tagging.

- Model: DistilBERT base uncased
- Architecture: Token classification
- Training samples: 480
- Validation samples: 120
- Training epochs: 3
- Training environment: CPU
- Exact parameter count: **66,375,953**
- Parameter constraint: Below 1 billion parameters

The model uses word-level labels aligned with tokenizer subwords. Special and non-first subword tokens are excluded from the loss calculation.

## PHI Categories Covered

The implementation targets the following identifier categories:

- Patient, relative and clinician names
- Locations, including street addresses and cities
- Healthcare facility names
- Calendar dates
- Ages above 89
- Telephone and fax numbers
- Email addresses
- Social Security Numbers (SSNs)
- Medical Record Numbers (MRNs)
- Health plan identifiers
- Account numbers
- Driver's licence identifiers
- URLs
- IP addresses

Relative time expressions such as "3 days post-op" are intended to remain unchanged.

## Dataset

No real patient data is used.

### Training Data

Synthetic clinical notes are generated using Faker and custom templates. The generated dataset contains 600 examples with token-level BIO annotations.

### Test Data

A separate set of 20 manually labelled synthetic clinical notes is used for evaluation.

The test dataset contains 47 labelled PHI entities.

Training and evaluation use separate datasets.

## Project Structure

```text
PHI-Deidentification-Gateway/
│
├── data/
│   ├── synthetic_train.json
│   └── synthetic_test.json
│
├── models/
│   └── distilbert_phi/
│
├── src/
│   ├── generate_data.py
│   ├── create_test_data.py
│   ├── detector.py
│   ├── deidentify.py
│   ├── rehydrate.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utility_evaluation.py
│   └── pipeline.py
│
├── tests/
│   └── test_pipeline.py
│
├── results/
│   ├── metrics.json
│   └── utility_evaluation.json
│
├── app.py
├── requirements.txt
├── README.md
├── FAILURES.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashiba713/phi-deidentification-gateway.git

cd phi-deidentification-gateway
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Install the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4. Generate synthetic training data

```bash
python src/generate_data.py
```

### 5. Train the model

```bash
python src/train.py
```

This generates the local model directory used by the detector.

### 6. Launch the application

Generate a Gemini API key using Google AI Studio:

https://aistudio.google.com/apikey

Start the application:

```bash
streamlit run app.py
```

Enter the API key in the application's sidebar.

**Security:** Never commit API keys or real patient information. The utility comparison script sends original notes to Gemini, so it must only be used with synthetic data.

## Usage

1. Enter a synthetic clinical note in the Streamlit interface.
2. The hybrid detector identifies PHI.
3. Detected identifiers are replaced with placeholders.
4. A local mapping is maintained.
5. Only the masked note is sent to Gemini.
6. Gemini generates a response using the placeholders.
7. The gateway restores identifiers locally.
8. The restored response is displayed.

### Example

Original note:

```text
Patient John Smith, aged 92, visited Green Valley Hospital.
```

Illustrative masked note:

```text
Patient [PERSON_1], aged [AGE_OVER_89_1],
visited [FACILITY_1].
```

Illustrative local mapping:

```json
{
  "[PERSON_1]": "John Smith",
  "[AGE_OVER_89_1]": "92",
  "[FACILITY_1]": "Green Valley Hospital"
}
```

The mapping remains local.

## Core Interface

The de-identification and restoration functions are exposed as:

```python
deidentify(text) -> (masked_text, mapping)

rehydrate(response, mapping) -> restored_response
```

Repeated occurrences of the same identifier within a note use the same placeholder.

## Evaluation

### Hybrid Detector vs Presidio Baseline

Evaluation was conducted using 20 manually labelled synthetic notes.

| Metric | Presidio baseline | Hybrid detector |
|---|---:|---:|
| Precision | 55.17% | 97.92% |
| Recall | 68.09% | 100% |
| F1-score | 60.95% | 98.95% |
| Leaked notes | 13/20 | 0/20 |
| Leak rate | 65% | 0% |

The hybrid detector improved recall and reduced missed identifiers on this synthetic test set.

### Evaluation Commands

Run the detector evaluation:

```bash
python src/evaluate.py
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

**Latest recorded test result:** 5 passed.

### LLM Utility Evaluation

Five synthetic notes were selected for comparison.

For each note:
1. Gemini generated a summary from the original note.
2. PHI was masked locally.
3. Gemini generated a summary from the masked note.
4. The response was rehydrated locally.
5. Original, masked and restored summaries were compared.

The experiment demonstrated that most clinical information remained understandable. It also identified a patient/clinician role reversal in one response and a duplicated MRN prefix in another.

Results are saved in:

```text
results/utility_evaluation.json
```

This is a qualitative five-note experiment, not a statistically validated clinical utility study.

## Privacy and Security Design

- Original-to-placeholder mappings remain local.
- Only masked text is sent to Gemini during normal application processing.
- API keys are entered at runtime rather than hardcoded.
- Synthetic data is used for training and testing.
- The utility experiment is restricted to synthetic notes.

The project is a prototype and has not undergone an independent security audit.

## HIPAA Safe Harbor Considerations

The implementation targets selected identifiers from the HIPAA Safe Harbor de-identification framework.

It does not claim to remove every identifier in the Safe Harbor list.

Categories not comprehensively implemented include:

- Biometric identifiers, including voice prints and fingerprints.
- Full-face photographs and comparable images.
- Vehicle identifiers and serial numbers.
- Device identifiers and serial numbers.
- Web URLs beyond the implemented URL pattern coverage.
- Employer-related identifiers.
- Other unique identifying characteristics or codes not covered by the current detector.

The implementation also does not establish that the resulting data is legally de-identified under HIPAA.

Reference: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html

## Limitations

- Synthetic data may not represent the complexity of real clinical documentation.
- The manually labelled test set contains only 20 notes.
- The utility evaluation contains only five notes.
- Model performance may vary across clinical domains and writing styles.
- Some identifiers may be missed or incorrectly classified.
- LLM-generated summaries may contain semantic errors.
- Rehydration restores placeholders but cannot correct LLM-generated factual or semantic mistakes.
- The application is not validated for production healthcare use.

## Future Improvements

- Expand the synthetic and independently annotated evaluation datasets.
- Evaluate on appropriately licensed clinical NER datasets.
- Improve contextual handling of ambiguous names and locations.
- Add stronger placeholder boundary and prefix handling.
- Expand coverage of the remaining HIPAA identifier categories.
- Conduct broader clinical utility and privacy evaluations.
- Add audit logging that does not expose original PHI.

## References

1. U.S. Department of Health and Human Services — HIPAA De-identification Guidance: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
2. Microsoft Presidio: https://microsoft.github.io/presidio/
3. Hugging Face Transformers: https://huggingface.co/docs/transformers/
4. DistilBERT: https://huggingface.co/distilbert/distilbert-base-uncased
5. Google AI Studio: https://aistudio.google.com/

## Disclaimer

This project is an educational and technical prototype. It is not a certified HIPAA-compliant system, does not guarantee complete PHI removal, and must not be used to process real patient information without appropriate validation, security controls and legal review.

## Author

**Ashiba Alben A**

AI & Data Science Engineering Student

GitHub: https://github.com/ashiba713





