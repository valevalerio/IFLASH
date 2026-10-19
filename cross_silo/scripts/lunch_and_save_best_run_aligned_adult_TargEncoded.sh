sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 100000 --num_clients 12 --batch_size=80 --class_weight=False --dropout=0.03887079828099323 --hidden_units=81 --layers=3 --learning_rate=0.004901989861685105 --local_epochs=2 --momentum=0.43664214891225345 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=80 --class_weight=False --dropout=0.03887079828099323 --hidden_units=81 --layers=3 --learning_rate=0.004901989861685105 --local_epochs=2 --momentum=0.43664214891225345 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 5 --num_clients 12 --batch_size=80 --class_weight=False --dropout=0.03887079828099323 --hidden_units=81 --layers=3 --learning_rate=0.004901989861685105 --local_epochs=2 --momentum=0.43664214891225345 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 100000 --num_clients 16 --batch_size=80 --class_weight=False --dropout=0.03887079828099323 --hidden_units=81 --layers=3 --learning_rate=0.004901989861685105 --local_epochs=2 --momentum=0.43664214891225345 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 5 --num_clients 16 --batch_size=80 --class_weight=False --dropout=0.03887079828099323 --hidden_units=81 --layers=3 --learning_rate=0.004901989861685105 --local_epochs=2 --momentum=0.43664214891225345 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 16 --batch_size=80 --class_weight=False --dropout=0.03887079828099323 --hidden_units=81 --layers=3 --learning_rate=0.004901989861685105 --local_epochs=2 --momentum=0.43664214891225345 --num_rounds=30 --no-sweep &
wait
