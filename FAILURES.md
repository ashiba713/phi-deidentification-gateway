# Development Failures and Lessons Learned

This document records issues encountered while developing the PHI De-identification Gateway and the changes made during development.

## 1. PHI Detection Errors

**Problem:** The initial hybrid detector produced false positives and missed certain identifiers, including medical record numbers, account identifiers, locations and ages over 89.

**Correction:** Added custom regular-expression patterns, refined entity filtering, and combined Presidio detections with DistilBERT predictions.

**Lesson:** Combining rule-based detection with a trained NER model improved detection coverage on the synthetic test set.

## 2. Relative Time Expressions

**Problem:** Relative expressions such as "3 days post-op" could be incorrectly identified as calendar dates.

**Correction:** Added filtering to preserve relative-duration expressions.

**Lesson:** Date detection must distinguish calendar dates from clinically meaningful relative time expressions.

## 3. Age Detection

**Problem:** The detector initially confused ordinary ages with dates and other numeric entities.

**Correction:** Added age-specific detection and filtering so that ages above 89 are targeted while ages 89 and below are preserved.

**Lesson:** Numeric context is important when identifying sensitive age information.

## 4. Location and Facility Detection

**Problem:** Partial addresses and hospital names were sometimes detected with incorrect boundaries. Some model predictions also included surrounding words or punctuation.

**Correction:** Improved custom location and facility patterns, added entity validation and refined overlapping-span handling.

**Lesson:** Correct entity boundaries matter for both precision and reliable restoration.

## 5. Overlapping Entity Predictions

**Problem:** Presidio, regular expressions and DistilBERT sometimes identified overlapping text spans.

**Correction:** Introduced detection prioritisation and non-overlapping span selection.

**Lesson:** Multiple detection sources require a consistent conflict-resolution strategy.

## 6. Gemini API Key Rejection

**Problem:** Gemini returned HTTP 403 because the API key was reported as leaked.

**Correction:** Replaced the rejected key with a newly generated API key.

**Lesson:** API keys must be kept private and must never be committed to source control.

## 7. Gemini Service Unavailability

**Problem:** Gemini temporarily returned HTTP 503 due to high demand.

**Correction:** Added retry handling for temporary server and rate-limit errors, with increasing wait intervals.

**Lesson:** External LLM services can be temporarily unavailable, so API integrations need error handling.

## 8. LLM Utility Evaluation Findings

**Problem:** In the five-note utility experiment, one generated summary reversed the roles of a patient and clinician. Another restored an MRN with a duplicated prefix.

**Correction:** These findings were documented for further improvement. They were not silently treated as successful semantic preservation.

**Lesson:** Rehydration can restore identifiers but cannot correct semantic errors introduced by an LLM. Placeholder replacement also requires careful handling of surrounding labels and prefixes.

## 9. Evaluation Limitations

- The training dataset is synthetic and programmatically generated.
- The manually labelled evaluation set contains 20 synthetic notes.
- The utility experiment contains five synthetic notes.
- Results may not generalise to real clinical documentation.
- The detector has not undergone clinical validation or independent security assessment.

## Conclusion

The development process involved iterative improvements to entity detection, span handling, API reliability and response restoration. Remaining limitations are documented rather than presented as solved problems.