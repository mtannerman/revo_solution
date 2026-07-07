## Generate data

[`generate_data.py`](generate_data.py)

### Prompt used

```
Please generate a dummy data set for me for a data labelling usecase into a csv. Please add the following columns:

* project: a string project identifier. 10 projects. Example: "project_2"
* job: a string job identifier within a project. 10 jobs. Example: "job_3"
* question: a string job identifier within a project/job. 10 questions. Example: "question_3"
* assessor: a string identifier of a labeller. 2k labellers. Example: "assessor_5"
* label: a binary label. There's only one vote per project/job/question/labeller.

Please for each question randomly (with a well-defined seed) determine the true label.
Please for each assessor randomly define the probability of agreeing with the true label.
the probability should be drawn from a uniform distrubition between [0.9, 1.]

For each label+assessor, draw from the above-mentioned assessor-specific distribution.

Save it into `data/data.csv`.

Please save the generator script to `generate_data.py`.
```

## Majority vote

[`get_majority_vote.py`](get_majority_vote.py)

### Prompt used

```
Please now write a Python script: `get_majority_vote.py`.

Task:

Please write a polars expression that gets from the data:
* the majority vote (label) by `project/job/question`. Please ensure that the tie mechanism is deterministic, alongside the whole query (maintain_order=True, etc).
* the standard deviation of the labels.
```

## Label quality

[`get_label_quality.py`](get_label_quality.py)

### Prompt used

```
Now please create another one: `get_label_quality.py`.

Please get the `project/job/question` counts by assessor.

Please identify the 5th percentile of it.

Please write a polars expression where you compute the agreement rate of the assessors.
The agreement rate should not count the `project/job/question` triplets for which there's only one assessor.
The agreement rate should not count the assessor's own labels.
You need to end up with a list of boolean agreement rates per assessor.
Please filter by the 5th percentile you computed before: filter out the ones who didn't contribute much yet.
Then, please apply a Bayesian estimate to the list of booleans per assessor to estimate the agreement rate of the assessor. Please use mean + std-dev as as the agreement estimate.

Please call this `assessor_quality`.

Then please use the assessor qualities to compute the weighted vote. You will need to remove the votes of the non-contributors, and compute a weighted average of the labels, where the weights are the agreement rates.
Please then cast the agreement rates to `pl.Int64()` to get the final label estimate.

Please call this `label_quality`.
```