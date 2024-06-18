## Diff-SR DrQ-v2 (Imgae-based POMDP)
### Installation & Usage
+ Python 3.10
    ```bash
    conda create --name <env> --file image_pomdp_deps.txt
    ```
+ Also install a freezed version of the metaworld
    ```bash
    git clone git@github.com:Farama-Foundation/Metaworld
    cd Metaworld && git checkout 04be337a
    pip install -e .
    ```
+ Run the code
    ```bash
    task=door-open
    python3 examples/train_image.py --config configs/latent_diff_sr.yaml --task "metaworld_$task"
    ```

