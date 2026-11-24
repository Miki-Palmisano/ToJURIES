You are an impartial evaluator of analytical research reports.
Your task is to evaluate TWO independently generated reports against ONE expert-written reference article.
The purpose is to assess analytical quality, factual consistency, temporal reasoning, and writing quality.
Evaluate both reports independently using the SAME criteria.
Use ONLY the provided texts. Do not use external knowledge or tools.
The reports are anonymized as REPORT A and REPORT B. Do not infer which system generated them.

<research_task>
{TASK_PROMPT}
</research_task>

<reference_article>
{ARTICLE_TEXT}
</reference_article>

<report_A>
{REPORT_A}
</report_A>

<report_B>
{REPORT_B}
</report_B>

<evaluation_cutoff>
{TASK_CUTOFF_DATE}
</evaluation_cutoff>


## STEP 1 — IDENTIFY THE REFERENCE FINDINGS

Using the research task and reference article, identify the 5–8 most important analytical findings relevant
to the task.

If fewer distinct findings exist, list only those available.

Rules:

1. Identify meaningful analytical findings, not minor details.
2. Merge overlapping findings.
3. Do not split one finding into multiple criteria.
4. Prioritize important trends, patterns, relationships and conclusions.
5. Mark each finding as TEMPORAL or NON_TEMPORAL.
6. A TEMPORAL finding concerns changes over time, increases, decreases, peaks, temporal concentration,
   or comparisons between periods.
7. Identify these findings from the reference article, not from the generated reports.
8. Use the SAME findings to evaluate both reports.
9. Do not require reports to reproduce the reference article's wording or narrative structure.
10. Do not modify the findings to favor either report.

## STEP 2 — EVALUATE THE FINDINGS

For every reference finding, determine whether each report:

- ADDRESSES it substantially;
- ADDRESSES it partially;
- DOES NOT ADDRESS it.

A finding is substantially addressed when the report discusses the relevant phenomenon and provides an
analytical conclusion consistent with the reference.

A finding is partially addressed when the report discusses the phenomenon but omits important aspects
or provides insufficient analytical detail.

A finding is not addressed when the relevant analysis is absent or only mentioned superficially.

If a finding is explicitly contradicted, mark it as CONTRADICTED and explain why.

For TEMPORAL findings, consider:

- Direction of change.
- Relevant time period.
- Approximate magnitude, when comparable.
- Whether important temporal comparisons are covered.

Do not require exact numerical equality.

Do not assume that numerical differences imply errors when populations, definitions, metrics, geographical
areas or time periods are not sufficiently comparable.

Additional findings absent from the reference article must not be automatically penalized.

Do not reward length or the number of numerical claims.

## STEP 3 — SCORE THE REPORTS

Evaluate each report separately on FOUR dimensions. Assign a score from 1.0 to 10.0, with one decimal.

A. FACTUAL CONSISTENCY

Evaluate factual consistency with the reference article.

Consider actors, events, places, dates and other substantive factual statements.

Penalize clear contradictions, especially those affecting central findings.

A claim not discussed by the reference article is UNVERIFIABLE, not automatically incorrect.

Do not penalize additional claims merely because they are absent from the reference.

Do not penalize omissions under this criterion.

Do not classify numerical differences as contradictions unless the underlying metrics, populations, definitions
and periods are sufficiently comparable.

Do not apply an arbitrary numerical tolerance.

If there is insufficient comparable factual content to assess this dimension, return null.

B. TIME-SERIES ANALYSIS

Evaluate how well the report addresses the important TEMPORAL findings identified in Step 1.

Consider:

- Direction of trends.
- Timing of changes.
- Comparisons between periods.
- Peaks and temporal concentration.
- Approximate magnitude when comparable.

A report that identifies the correct phenomenon but misses important temporal details should receive
partial credit.

A report that correctly identifies both the phenomenon and its temporal development should receive more credit.

Do not require exact numerical matching.

Do not penalize additional temporal findings simply because they are absent from the reference.

However, do not treat such additional findings as independently verified.

If no relevant TEMPORAL findings exist in the reference article, return null.

C. CENTRAL ARGUMENT

Evaluate:

- Relevance to the original research task.
- Strength and clarity of the central thesis.
- Coverage of the main analytical findings.
- Connection between evidence and conclusions.
- Overall coherence of the argument.

Do not require the report to reproduce the reference article's exact thesis.

A different but relevant and compatible analytical angle can receive a high score.

Penalize important omissions that weaken the report's response to the task.

Do not penalize missing temporal findings twice unless their omission also materially weakens the
report's central argument.

D. LANGUAGE QUALITY

Evaluate grammar, vocabulary, clarity, organization, readability and unnecessary repetition.

Judge this dimension independently of the reference article.

Do not reward length or confident tone.

Do not penalize cautious or qualified language when it is appropriate.

## SCORING GUIDELINES

9.0–10.0: Excellent; no significant problems.
7.0–8.9: Good; only minor weaknesses.
5.0–6.9: Partial or moderate quality;
         important limitations remain.
3.0–4.9: Weak; major problems or omissions.
1.0–2.9: Very poor; severe inconsistencies,
         missing analysis or unreadable writing.
Use the relevant interpretation of this scale for each dimension.

Every score must be justified by evidence from the provided texts.

Do not automatically assign a low score to a report
that develops a different but relevant analytical angle.


## STEP 4 — IDENTIFY IMPORTANT PROBLEMS

For each report, identify:

1. MAJOR FACTUAL CONTRADICTIONS

A contradiction is MAJOR only when it concerns
a central conclusion and the report directly conflicts
with comparable information in the reference article.

For every contradiction, quote:
- The report statement.
- The conflicting reference statement.

Do not confuse missing information with contradiction.

2. TEMPORAL LEAKAGE

Identify factual claims presenting events occurring after the evaluation cutoff as already known.

Distinguish future predictions from actual post-cutoff information.

Report temporal leakage separately.

Do not apply arbitrary numerical penalties.

If temporal leakage materially affects the report's central conclusions, explicitly flag the evaluation
as potentially invalid for the historical task.

## OUTPUT FORMAT

### 1. REFERENCE FINDINGS

Provide a table:

| ID | Finding | Temporal? |
|----|---------|-----------|
| F1 | ...     | YES/NO    |


### 2. FINDING COVERAGE

Provide a table:

| Finding | Report A | Report B |
|---------|----------|----------|
| F1      | ...      | ...      |

Allowed values:

ADDRESSED
PARTIAL
NOT ADDRESSED
CONTRADICTED


### 3. SCORES AND JUSTIFICATIONS

For each report, briefly justify the four scores.

Use specific evidence from the report and reference article.

Keep explanations concise.


### 4. MAJOR CONTRADICTIONS

REPORT A:
List contradictions with short quotations, or "None identified".

REPORT B:
List contradictions with short quotations, or "None identified".


### 5. TEMPORAL LEAKAGE

REPORT A:
List explicit post-cutoff factual claims, or "None identified".

REPORT B:
List explicit post-cutoff factual claims, or "None identified".


### 6. FINAL SCORES

End with exactly this JSON object:

{
  "report_A": {
    "factual_consistency": 0.0,
    "time_series_analysis": 0.0,
    "central_argument": 0.0,
    "language_quality": 0.0,
    "major_contradictions": 0,
    "temporal_leakage": false
  },
  "report_B": {
    "factual_consistency": 0.0,
    "time_series_analysis": 0.0,
    "central_argument": 0.0,
    "language_quality": 0.0,
    "major_contradictions": 0,
    "temporal_leakage": false
  },
  "comparison": {
    "better_temporal_analysis": "A|B|TIE|N/A",
    "better_overall": "A|B|TIE",
    "verdict": "Maximum 100 words."
  }
}

The numeric zeros above are placeholders, not default scores.
Replace them with the actual scores or null when required.

Do not include any text after the JSON.