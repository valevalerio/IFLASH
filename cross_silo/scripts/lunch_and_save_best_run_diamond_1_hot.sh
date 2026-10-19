sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 100000 --num_clients 12 --layers 2 --num_rounds 30 --batch_size=4 --class_weight=True --dropout=0.005078161945682225 --hidden_units=62 --learning_rate=0.06230386285876934 --local_epochs=3 --momentum=0.8924320788060621 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.99 --num_clients 12 --layers 2 --num_rounds 20 --batch_size=16 --class_weight=False --dropout=0.0015510438204883626 --hidden_units=56 --learning_rate=0.24598978440270244 --local_epochs=4 --momentum=0.3743129516741612 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 5 --num_clients 12 --batch_size=64 --class_weight=False --dropout=0.028699374159479832 --hidden_units=87 --layers=1 --learning_rate=0.23457167821210995 --local_epochs=3 --momentum=0.15103037599897937 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 5 --num_clients 16 --batch_size=32 --class_weight=False --dropout=0.06628214555577164 --hidden_units=89 --layers=2 --learning_rate=0.05806964482262788 --local_epochs=4 --momentum=0.269871972218224 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 100000 --num_clients 16 --batch_size=8 --class_weight=True --dropout=0.022737868176898247 --hidden_units=123 --layers=3 --learning_rate=0.14075338974402735 --local_epochs=5 --momentum=0.30103249607404975 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.99 --num_clients 16 --batch_size=80 --class_weight=False --dropout=0.01374413579384286 --hidden_units=86 --layers=2 --learning_rate=0.2025665317725958 --local_epochs=5 --momentum=0.1387020576871172 --num_rounds=10 --no-sweep &
wait
