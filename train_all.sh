steps=100000

for dataset in MNIST; do
    for model in ALI VAEGAN VAE AE; do
        python train.py $dataset"_"$model $steps
    done
done

