echo Sweep_adult_TargEncoded_12_imb_100000 
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 100000 --num_clients 12 --batch_size=16 --class_weight=False --dropout=0.006345551334688615 --hidden_units=45 --layers=3 --learning_rate=0.16549640096290782 --local_epochs=3 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &> adult_TargEncoded_12_imb_100000.out &
wait
echo Sweep_adult_TargEncoded_12_imb_0.99
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 12 --batch_size=16 --class_weight=False --dropout=0.006345551334688615 --hidden_units=45 --layers=3 --learning_rate=0.16549640096290782 --local_epochs=3 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &> adult_TargEncoded_12_imb_0.99.out &
wait
echo Sweep_adult_TargEncoded_12_imb_5
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 5.0 --num_clients 12 --batch_size=16 --class_weight=False --dropout=0.006345551334688615 --hidden_units=45 --layers=3 --learning_rate=0.16549640096290782 --local_epochs=3 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &> adult_TargEncoded_12_imb_5.out &
wait

python3 sim.py --dataset_name adult_TargEncoded --alpha 100000 --num_clients 16 --batch_size=16 --class_weight=False --dropout=0.006345551334688615 --hidden_units=45 --layers=3 --learning_rate=0.16549640096290782 --local_epochs=3 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &> adult_TargEncoded_16_imb_100000.out &
wait
echo Sweep_adult_TargEncoded_12_imb_0.99
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 0.99 --num_clients 16 --batch_size=16 --class_weight=False --dropout=0.006345551334688615 --hidden_units=45 --layers=3 --learning_rate=0.16549640096290782 --local_epochs=3 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &> adult_TargEncoded_16_imb_0.99.out &
wait
echo Sweep_adult_TargEncoded_12_imb_5
sleep 3
python3 sim.py --dataset_name adult_TargEncoded --alpha 5.0 --num_clients 16 --batch_size=16 --class_weight=False --dropout=0.006345551334688615 --hidden_units=45 --layers=3 --learning_rate=0.16549640096290782 --local_epochs=3 --momentum=0.1325989715500346 --num_rounds=30 --no-sweep &> adult_TargEncoded_16_imb_5.out &
