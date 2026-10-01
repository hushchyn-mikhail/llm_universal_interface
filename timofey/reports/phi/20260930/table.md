# Phi results

ROC-AUC point estimate ± test-bootstrap standard deviation. No training-seed uncertainty is implied.

Status counts: completed=43, failed=5, missing=8.

Blank entries are unavailable or excluded; they are never replaced with 0.5.

| Dataset | Zero shot | Few shot | Tuned 0% missing | Tuned 20% missing | Tuned 50% missing | Tuned 90% missing | Multitask |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bank | 0.5676 ± 0.0107 | 0.6373 ± 0.0103 | 0.9222 ± 0.0037 | 0.8955 ± 0.0046 | 0.8257 ± 0.0068 | 0.6343 ± 0.0091 | — (missing) |
| blood | 0.6962 ± 0.0560 | 0.6166 ± 0.0559 | 0.7692 ± 0.0499 | 0.7635 ± 0.0474 | — (failed) | 0.5322 ± 0.0538 | — (missing) |
| california | 0.5792 ± 0.0087 | 0.6776 ± 0.0082 | — (failed) | 0.9179 ± 0.0040 | 0.8470 ± 0.0057 | — (failed) | — (missing) |
| car | 0.6154 ± 0.0184 | 0.9467 ± 0.0074 | — (failed) | 0.9679 ± 0.0060 | — (failed) | 0.6717 ± 0.0292 | — (missing) |
| credit_g | 0.4936 ± 0.0412 | 0.5381 ± 0.0404 | 0.7408 ± 0.0383 | 0.7041 ± 0.0391 | 0.6689 ± 0.0417 | 0.5758 ± 0.0433 | — (missing) |
| diabetes | 0.8151 ± 0.0337 | 0.7067 ± 0.0402 | 0.8450 ± 0.0310 | 0.8063 ± 0.0352 | 0.6918 ± 0.0407 | 0.5522 ± 0.0480 | — (missing) |
| heart | 0.7911 ± 0.0361 | 0.8180 ± 0.0320 | 0.9401 ± 0.0180 | 0.8995 ± 0.0237 | 0.8452 ± 0.0338 | 0.6937 ± 0.0400 | — (missing) |
| income | 0.8010 ± 0.0052 | 0.8276 ± 0.0046 | 0.9295 ± 0.0027 | 0.9184 ± 0.0029 | 0.8806 ± 0.0036 | 0.6956 ± 0.0059 | — (missing) |
