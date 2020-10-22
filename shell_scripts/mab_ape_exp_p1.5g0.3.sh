param_ucb=0.03
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ucb-truncated-mean --param $param_ucb --seed 40

param_dsee=0.01
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo dsee --param $param_dsee --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo dsee --param $param_dsee --seed 40

param_gsr=0.02
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo gsr --param $param_gsr --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo gsr  --param $param_gsr --seed 40

param_gamma=0.005
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-gamma --param $param_gamma --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-gamma --param $param_gamma --seed 40

param_GEV=0.005
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-GEV --param $param_GEV --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-GEV  --param $param_GEV --seed 40

param_frechet=0.005
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-frechet --param $param_frechet --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-frechet --param $param_frechet --seed 40

param_pareto=0.005
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-pareto --param $param_pareto --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --moment_param 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo ape-pareto --param $param_pareto --seed 40
