# To_JURIES

Confronta due report (es. PRE vs POST) con giudici LLM (GPT, Gemini) rispetto a un articolo di riferimento, usando **il tuo prompt di valutazione, senza modifiche** (`to_juries/prompts/evaluation.md`). Vengono sostituiti solo i cinque segnaposto (`{TASK_PROMPT}`, `{ARTICLE_TEXT}`, `{REPORT_A}`, `{REPORT_B}`, `{TASK_CUTOFF_DATE}`).

## Uso

`evaluate` chiama ogni giudice su ogni coppia in ogni ordine e scrive in `out_dir`:
`config.json` (giudici senza chiavi, opzioni, hash del prompt), `prompt.md`, `runs/<giudice>/<coppia>__<ordine>__r0.json` (risposta completa e JSON letto, uno per chiamata), `results.json` (tutti i verdetti, con il sistema dietro REPORT A/B).
Le chiamate riuscite sono in cache: rilanciando sulla stessa cartella si ripetono solo gli errori (`force=True` per rifare tutto, `limit=2` per una prova). `rj.analyze("results/exp1")` rianalizza dalla sola cartella.

Opzioni di `Evaluator`: `systems`, `orders`, `reps`, `concurrency`, `max_retries`, `bootstrap`, `alpha`, `seed`. Per un altro modello basta una sottoclasse di `rj.Provider` con il metodo `complete(prompt)`.

Il prompt deve contenere `{TASK_PROMPT}`, `{ARTICLE_TEXT}`, `{REPORT_A}`, `{REPORT_B}`, `{TASK_CUTOFF_DATE}` e chiedere il JSON finale con `report_A`/`report_B`; vengono sostituiti solo questi segnaposto.

## CSV
`pair_id, task_prompt (o task_file), reference_file, cutoff, baseline_file, candidate_file`

## Output (`results/`)
`report.md`, `summary.json`, `scores_long.csv`, `pair_gains.csv`, `per_judge.csv`, `jury.csv`, `order_effects.csv`, `agreement.csv`, `flags.csv`.

## Note
- Le chiamate ai provider usano le SDK ufficiali (`openai` con `AzureOpenAI`, `google-genai`) ma non sono state provate con API reali; il resto è testato con un provider finto (`pytest`).
- Se manca una chiave o un parametro Azure/Gemini, il giudice lancia `ConfigError` subito alla creazione.
- Se il JSON finale non è valido il giudice viene richiamato (fino a `max_retries`).
