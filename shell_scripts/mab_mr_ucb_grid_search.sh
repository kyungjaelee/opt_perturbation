for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.5 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.5 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.2 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.5 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.5 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.5 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.1 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.3 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.5 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.5 --samples 20000 --arms 10 --algo mr-ucb --seed 10

for i in {1..9}
do
    python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 10 --algo mr-ucb --seed $i &
done
python3 -m multi_armed_bandit.run_grid_search --noise pareto --moment 1.8 --scale 1.0 --mean one_hot --gap 0.7 --samples 20000 --arms 10 --algo mr-ucb --seed 10