sleep 3
python3 sim.py --dataset_name dry_bean --alpha 100000 --num_clients 12 --layers 2 --num_rounds 30 --batch_size=8 --class_weight=False --dropout=0.18686979048958924 --hidden_units=28 --learning_rate=0.12443168739837268 --local_epochs=5 --momentum=0.4813496691137441 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name dry_bean --alpha 100000 --num_clients 16 --batch_size=128 --class_weight=False --dropout=0.03482907120606822 --hidden_units=33 --layers=2 --learning_rate=0.3384896566603923 --local_epochs=4 --momentum=0.07507789639317923 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name dry_bean --alpha 0.99 --num_clients 12 --layers 2 --num_rounds 30 --batch_size=64 --class_weight=False --dropout=0.05966957654591041 --hidden_units=124 --learning_rate=0.08918890682655946 --local_epochs=2 --momentum=0.8211109463796185 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name dry_bean --alpha 0.99 --num_clients 16 --batch_size=80 --class_weight=True --dropout=0.04949338501193743 --hidden_units=109 --layers=1 --learning_rate=0.12172479750865754 --local_epochs=4 --momentum=0.6478321346320393 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name dry_bean --alpha 5 --num_clients 12 --batch_size=80 --class_weight=False --dropout=0.0925713988236753 --hidden_units=56 --layers=3 --learning_rate=0.04304536720136045 --local_epochs=2 --momentum=0.7547086036633756 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name dry_bean --alpha 5 --num_clients 16 --batch_size=4 --class_weight=True --dropout=0.17488814276957287 --hidden_units=23 --layers=1 --learning_rate=0.13491393909480504 --local_epochs=4 --momentum=0.8028497867626486 --num_rounds=30 --no-sweep &
wait
sleep 3
python3 sim.py --dataset_name dry_bean --alpha 10000 --num_clients 16 --batch_size=16 --class_weight=False --dropout=0.002735537690544421 --hidden_units=11 --layers=1 --learning_rate=0.2704752641990809 --local_epochs=5 --momentum=0.16186395961618347 --num_rounds=30 --no-sweep &
wait
