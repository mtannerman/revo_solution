import csv
import os
import random

SEED = 42
rng = random.Random(SEED)

N_PROJECTS = 10
N_JOBS = 10
N_QUESTIONS = 10
N_ASSESSORS = 2000

# True label per project/job/question
true_labels = {
    (p, j, q): rng.randint(0, 1)
    for p in range(N_PROJECTS)
    for j in range(N_JOBS)
    for q in range(N_QUESTIONS)
}

# Per-assessor probability of agreeing with the true label, uniform in [0.9, 1.0]
assessor_accuracy = {a: rng.uniform(0.9, 1.0) for a in range(N_ASSESSORS)}

out_path = os.path.join("data", "data.csv")
os.makedirs("data", exist_ok=True)

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["project", "job", "question", "assessor", "label"])
    for (p, j, q), true_label in true_labels.items():
        for a in range(N_ASSESSORS):
            if rng.random() < assessor_accuracy[a]:
                label = true_label
            else:
                label = 1 - true_label
            writer.writerow([f"project_{p}", f"job_{j}", f"question_{q}", f"assessor_{a}", label])

print(f"Wrote {out_path}: {N_PROJECTS * N_JOBS * N_QUESTIONS * N_ASSESSORS} rows")
