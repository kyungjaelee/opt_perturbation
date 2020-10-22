param_ucb=0.01
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.8 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.8 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed 20

param_ucb=0.01
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.8 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.8 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed 40