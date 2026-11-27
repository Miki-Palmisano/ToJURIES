import to_juries as tj

# Forma la Giuria
judges = [
    tj.AzureOpenAI(deployment="gpt-6-astra", temperature=None),
    tj.Gemini(model="gemini-4-argon"),
]

# Definisci il prompt
prompt = tj.Prompt.default()

# Inizializzi le coppie
#data = tj.PairDataset.from_json("config.json")
data = tj.PairDataset.from_records([
    {
        "pair_id": "p01",
        "cutoff": "2024-12-12",
        "task_prompt": "Do a deep research on Petro's presidency in Columbia and Columbia's conflict summary and outlook, as if we are at the end of 2024 heading into 2025",
        "reference_file": "../data/p01/reference.md", #or reference_text for text inline
        "first_file":"../data/p01/report_a.md", #or first_text for text inline
        "second_file":"../data/p01/report_b.md" #or second_text for text inline
    }
])

# Imposta la giuria con i parametri
juries = tj.Evaluator(judges, prompt, systems=("PRE", "POST"), orders=("AB", "BA"))

# Avvia la valutazione
run = juries.evaluate(data, out_dir="results/exp1")   # crea la cartella con i JSON, token richiesti
res = run.analyze()                                         # opzionale: restituisce dettagli statistici sulle valutazioni, no token richiesti
es = run.analyze()
print(res["summary"])     # media e overall per giudice e giuria
print(res["gain_stats"])  # statistiche sul guadagno (la giuria ha anche ci_low_boot / ci_high_boot)
print(res["agreement"])   # accordo tra i giudici
print(res["flags"])       # contraddizioni e leakage
print(res["votes"])       # conteggi di better_overall