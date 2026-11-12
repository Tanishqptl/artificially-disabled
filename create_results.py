import pandas as pd
from pathlib import Path

questions = pd.read_csv("evaluation/evaluation_questions.csv")

results = questions[["question_id", "question_type", "question"]].copy()
results["model_answer"] = ""
results["retrieved_citations"] = ""
results["correctness_score"] = ""
results["citation_precision"] = ""
results["abstained_correctly"] = ""
results["latency_seconds"] = ""
results["notes"] = ""

results.to_csv("evaluation/evaluation_results.csv", index=False)

print("Created evaluation/evaluation_results.csv")
print("Questions:", len(results))