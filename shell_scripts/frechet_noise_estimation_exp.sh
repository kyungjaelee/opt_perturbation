for i in {1..19}
do
    python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
done
python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed 20

#for i in {31..59}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed 60
#for i in {61..89}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed 90
#for i in {91..99}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.2 --scale 1.0 --samples 100000 --steps 1000 --seed 100

for i in {1..19}
do
    python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
done

#python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed 20
#for i in {31..59}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed 60
#for i in {61..89}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed 90
#for i in {91..99}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.5 --scale 1.0 --samples 100000 --steps 1000 --seed 100

for i in {1..19}
do
    python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
done
python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed 20

#for i in {31..59}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed 60
#for i in {61..89}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed 90
#for i in {91..99}
#do
#    python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed $i &
#done
#python3 -m estimation.run_experiments.py --noise frechet --moment 1.8 --scale 1.0 --samples 100000 --steps 1000 --seed 100

