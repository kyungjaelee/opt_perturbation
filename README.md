Bandits
---
1. Single run
```
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.1 --scale 1.0 --mean one_hot --gap 0.1 --samples 30000 --arms 100 --algo ape-pareto --seed 1
```
* --noise: pareto
* --moment: 1.1 
* --scale: 1.0 
* --mean: one_hot 
* --gap: 0.1 
* --samples: 30000 
* --arms: 100 
* --algo: ape-pareto
** Possible algorithms: 
** 'ape-weibull', 'ape-frechet', 'ape-pareto', 'ape-gamma', 'ape-GEV', 'ape-bounded', 'ucb-truncated-mean', 'ucb-catoni-mean', 'ucb-median-of-mean', 'mr-ucb', 'dsee', 'gsr'
** 'ape-bounded' and 'mr-ucb' are the proposed method.
* --seed: 1

2. Multiple runs with shell script
```
./shell_scripts/mab_ape_grid_search_p1.2g0.1.sh
```
* The shell script parallely runs 20 seeds per each algorithm
* By using the following script,
```
for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.1 --samples 5000 --arms 10 --algo ucb-truncated-mean --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.1 --samples 5000 --arms 10 --algo ucb-truncated-mean --seed 10
```
* See ./shell_scripts/mab_ape_grid_search_p1.2g0.1.sh for more detail
