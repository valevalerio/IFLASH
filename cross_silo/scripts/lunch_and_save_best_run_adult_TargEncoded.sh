echo Sweep_adult_TargEncoded_12_imb_100000
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
echo Sweep_adult_TargEncoded_12_imb_5
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
echo Sweep_adult_TargEncoded_16_imb_100000
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
echo Sweep_adult_TargEncoded_08_imb_5
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
echo Sweep_adult_TargEncoded_16_imb_5
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
echo Sweep_adult_TargEncoded_08_imb_0.99
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
echo Sweep_adult_TargEncoded_16_imb_0.99
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
echo Sweep_adult_TargEncoded_08_imb_0.50
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=8 --class_weight=True --dropout=0.004044936402889565 --hidden_units=45 --layers=3 --learning_rate=0.25652605064825623 --local_epochs=5 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &
wait
