param_GEV=0.001
for i in {1..19}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-GEV --param $param_GEV --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-GEV  --param $param_GEV --seed 20

param_GEV=0.001
for i in {21..39}
do
    python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-GEV --param $param_GEV --seed $i &
done
python3 -m multi_armed_bandit.run_experiments --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo ape-GEV  --param $param_GEV --seed 40