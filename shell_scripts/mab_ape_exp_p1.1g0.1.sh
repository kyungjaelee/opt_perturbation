param_ucb = 0.02
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed 20

param_dsee = 0.02
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo dsee --param $param_dsee --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo dsee --param $param_dsee --seed 20

param_gsr = 0.02
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo gsr --param $param_gsr --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo gsr  --param $param_gsr --seed 20

param_gamma = 0.02
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-gamma --param $param_gamma --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-gamma --param $param_gamma --seed 20

param_GEV = 0.02
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-GEV --param $param_GEV --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-GEV  --param $param_GEV --seed 20

param_frechet = 0.02
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-frechet --param $param_frechet --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-frechet --param $param_frechet --seed 20

param_pareto = 0.02
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-pareto --param $param_pareto --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.1 --scale 0.1 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-pareto --param $param_pareto --seed 20