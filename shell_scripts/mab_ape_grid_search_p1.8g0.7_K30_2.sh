for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ucb-truncated-mean --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ucb-truncated-mean --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ucb-catoni-mean --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ucb-catoni-mean --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ucb-median-of-mean --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ucb-median-of-mean --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo dsee --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo dsee --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo gsr --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo gsr --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-gamma --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-gamma --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-GEV --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-GEV --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-pareto --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-pareto --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-weibull --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-weibull --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-frechet --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-frechet --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-uniform --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-uniform --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-rademacher --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-rademacher --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-rademacher2 --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-rademacher2 --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-rademacher3 --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo ape-rademacher3 --seed 20

for i in {11..19}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 30 --algo mr-ucb --seed 20
