# Data

The annotated corpus (~9,000 tweets) is **not redistributed** here because of Twitter/X
terms of service and privacy restrictions. `example.csv` shows the expected format only
— it is not real data.

## Format

UTF-8 CSV (a BOM is fine) with two required columns:

| column  | type | description                          |
|---------|------|--------------------------------------|
| `Tweet` | str  | raw tweet text                       |
| `Label` | int  | `0` Neither · `1` Abusive-only · `2` Hate-speech |

Put your splits here as `train.csv`, `val.csv`, `test.csv` (ignored by git).

## Corpus summary

- Tweets retrieved via the Twitter API with a hand-built lexicon of Chinese slurs and
  hate-speech keywords (see the [baseline repo](https://github.com/chingachleung/Chinese_Hate_Speech-Baseline-)).
- Mixed Simplified Chinese, Traditional Chinese and written Cantonese.
- Imbalanced: *Neither* is the majority class, so training uses class-weighted loss
  and model selection uses macro-F1.
- **Abusive-only** = offensive or profane but not targeting a group on a protected
  characteristic; **Hate-speech** = attacks or dehumanises a group (ethnicity,
  nationality, religion, gender, sexual orientation, etc.).
