echo Sweep_diamond_1_hot_16_imb_5
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 5 --num_clients 16 --batch_size=32 --class_weight=False --dropout=0.06628214555577164 --hidden_units=89 --layers=2 --learning_rate=0.05806964482262788 --local_epochs=4 --momentum=0.269871972218224 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_08_imb_5
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 5 --num_clients 8 --batch_size=16 --class_weight=True --dropout=0.023368638967926916 --hidden_units=23 --layers=1 --learning_rate=0.1035501513483028 --local_epochs=1 --momentum=0.607164174002604 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_04_imb_5
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 5 --num_clients 4 --batch_size=4 --class_weight=False --dropout=0.04129343297419435 --hidden_units=47 --layers=3 --learning_rate=0.10636375132569868 --local_epochs=2 --momentum=0.5150615034849348 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_04_iid
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 100000 --num_clients 4 --batch_size=64 --class_weight=False --dropout=0.029775183465150557 --hidden_units=55 --layers=3 --learning_rate=0.1790640859082115 --local_epochs=4 --momentum=0.29651186774144045 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_08_iid
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 100000 --num_clients 8 --batch_size=128 --class_weight=True --dropout=0.008945458601808154 --hidden_units=57 --layers=3 --learning_rate=0.1059537712870148 --local_epochs=5 --momentum=0.14317278363515876 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_16_iid
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 100000 --num_clients 16 --batch_size=32 --class_weight=False --dropout=0.007417055813902351 --hidden_units=98 --layers=2 --learning_rate=0.0997864784311446 --local_epochs=4 --momentum=0.7506309246372638 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_04_imb_0.99
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.99 --num_clients 4 --batch_size=128 --class_weight=False --dropout=0.011373669798189812 --hidden_units=25 --layers=3 --learning_rate=0.06496202923751948 --local_epochs=5 --momentum=0.6243692552566665 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_08_imb_0.99
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.99 --num_clients 8 --batch_size=128 --class_weight=False --dropout=0.006027130487030075 --hidden_units=68 --layers=2 --learning_rate=0.389248191349566 --local_epochs=2 --momentum=0.1566429862585006 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_16_imb_0.99
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.99 --num_clients 16 --batch_size=80 --class_weight=False --dropout=0.01374413579384286 --hidden_units=86 --layers=2 --learning_rate=0.2025665317725958 --local_epochs=5 --momentum=0.1387020576871172 --num_rounds=10 --no-sweep &
wait
echo Sweep_diamond_1_hot_04_imb_0.50
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.5 --num_clients 4 --batch_size=64 --class_weight=False --dropout=0.0018600394498341057 --hidden_units=70 --layers=3 --learning_rate=0.07617821989545784 --local_epochs=5 --momentum=0.37868969154359133 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_08_imb_0.50
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.5 --num_clients 8 --batch_size=128 --class_weight=False --dropout=0.1961826804016081 --hidden_units=69 --layers=2 --learning_rate=0.1752766803895051 --local_epochs=4 --momentum=0.2544981827244126 --num_rounds=30 --no-sweep &
wait
echo Sweep_diamond_1_hot_16_imb_0.50
sleep 3
python3 sim.py --dataset_name diamond_1_hot --alpha 0.5 --num_clients 16 --batch_size=80 --class_weight=True --dropout=0.18360287736780112 --hidden_units=42 --layers=3 --learning_rate=0.10351658052638602 --local_epochs=4 --momentum=0.4491303566184224 --num_rounds=30 --no-sweep &
wait
