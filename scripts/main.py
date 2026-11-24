import to_juries as tj

# Forma la Giuria
judges = [
    tj.AzureOpenAI(deployment="gpt-6-astra", temperature=None),
    tj.Gemini(model="gemini-4-argon"),
]

# Definisci il prompt
prompt = tj.Prompt.default()

# Inizializzi le coppie
data = tj.PairDataset.from_csv("coppie.csv")

# Avvia la valutazione
juries = tj.Evaluator(judges, prompt, systems=("PRE", "POST"), orders=("AB", "BA"))
run = juries.evaluate(data, out_dir="results/exp1")   # crea la cartella con i JSON
run.analyze()                                            # opzionale: report.md e tabelle