for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ucb-truncated-mean --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ucb-truncated-mean --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ucb-catoni-mean --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ucb-catoni-mean --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ucb-median-of-mean --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ucb-median-of-mean --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo dsee --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo dsee --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo gsr --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo gsr --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-gamma --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-gamma --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-GEV --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-GEV --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-pareto --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-pareto --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-weibull --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-weibull --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-frechet --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-frechet --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-uniform --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-uniform --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-rademacher --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo ape-rademacher --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 5000 --arms 10 --algo mr-ucb --seed 10
