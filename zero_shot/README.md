# Run Experiments


```
./run.sh
```
Note: Please change `run.sh` accordingly before running; check the script for more details.

Define your strategy before running an experiment. There are currently 3 strategies:

    - raw

    - rank_rrf

    - rank_normal 

raw: Only use the raw scores.

rank_rrf: Reciprocal Rank Fusion.

rank_normal: Ranking only.

run_name: to specify the name of the log file.

# Experiment outputs names
Whene finishing an experiment, outputs will be written in the same folder as `model_predict.py` file. It has the format: 
`logs_{proxy}_{run_name}.json`
