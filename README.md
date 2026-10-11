# DefaultRadar: MSME Default Prediction, 12 Months Ahead

**IDBI Innovate 2026 · Track 04 (Default Prediction Model)**

A probability-of-default model that flags stressed MSME loans **12 months in
advance** by fusing **structured data** (repayment behaviour, GST trend,
Account-Aggregator bank-feed signals) with **unstructured data** (field-visit
and call-note flags, which this prototype simulates as stand-ins for a
planned production NLP layer), and reports through **one common
risk-decile framework across all MSME loan types**: Term, Working Capital,
Mudra and Machinery.

## The problem

The bank's current early-warning capability is **16–22% accurate**, uses only
structured data, and is fragmented across loan types and borrower segments.

## Prototype results (synthetic panel, 12,000 loans, 25% held-out test)

| Metric | DefaultRadar (GBM) | Logistic baseline |
|---|---|---|
| Accuracy @ 0.5 | **91.8%** | 91.6% |
| ROC-AUC | 0.891 | 0.895 |
| Default recall | **55.9%** | 51.5% |
| KS statistic | 0.648 | 0.651 |

Accuracy is measured against a 13.5% default base rate: predicting "no
default" for every loan would already score 86.5% on this test set.

- **Top 3 risk deciles capture 85% of all defaults**: collections and
  restructuring effort can be focused on 30% of the book.
- **Consistent across loan types**: AUC 0.876–0.909 for Machinery, Mudra,
  Term and Working-Capital loans from a single model, which is the common
  interpretation framework the problem statement asks for.
- The `note_*` features (negative sentiment, dispute mentions, inventory
  pile-up) are simulated numeric flags, not text: `train.py` draws them from
  the same latent stress variable that generates the default label, as
  stand-ins for an NLP layer planned for production. Their importance
  (dispute flag 4th, negative sentiment 7th of 13 features) is built into
  the generator, so this synthetic setup cannot validate the
  structured + unstructured fusion approach.

## Run it

```bash
pip install numpy scikit-learn
python train.py        # trains both models, writes results.json + report/index.html
```

Open `report/index.html` for the full benchmark dashboard (ROC, permutation
importance, decile lift, per-loan-type consistency).

## Production architecture (post-shortlist)

- Feature store fed by LMS/CBS repayment events, GSTN pulls, AA consented
  bank feeds, and an NLP pipeline (IndicBERT-class embeddings) over field
  notes and call transcripts.
- Monthly scoring batch + event-triggered rescoring on bounce/DPD events;
  champion–challenger governance, SHAP-based reason codes for every alert,
  drift monitoring.
- Output: EWS deciles into collections workflow with recommended actions
  (restructure / enhanced monitoring / limit review).

---
*All data is synthetic, generated inside `train.py`. No real borrower data.*
