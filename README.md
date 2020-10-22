Bandits
---
1. Single run
```
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.1 --scale 1.0 --mean one_hot --gap 0.1 --samples 30000 --arms 100 --algo ape-pareto --seed 1
```
* --noise: pareto (fixed)
* --moment: 1.1 (fixed)
* --scale: 1.0 (fixed)
* --mean: one_hot (fixed)
* --gap: 0.1 (fixed)
* --samples: 30000 (fixed)
* --arms: 100 (fixed)
* --algo: ape-pareto (change this to run different algorithms such as 'ucb-truncated-mean', 'ucb-catoni-mean', and 'ucb-median-of-mean')
* --seed: 1 (set any seed between 1 ~ 20)
2. Multiple runs with shell script
```
./shell_scripts/mab_ape_grid_search.sh
```
* The shell script parallely runs 20 seeds per each algorithm
* See ./shell_scripts/mab_ape_grid_search.sh for more detail
