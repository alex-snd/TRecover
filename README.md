<h1 align="center">Welcome to Text Recovery Project 👋</h1>
<p align="center">
  A python library for training a Transformer neural network to solve the <a href="https://en.wikipedia.org/wiki/Running_key_cipher">Running Key Cipher</a>, widely known in the field of cryptography.
</p>

![Preview Animation](../assets/preview_animation.gif?raw=true)
  
<p align="center">
  <a href="https://huggingface.co/spaces/alex-snd/TRecover">
    <img src="https://img.shields.io/badge/demo-%F0%9F%A4%97%20Hugging%20Face-blue?color=%2348466D" alt="Hugging Face demo"/>
  </a>
  <a href="https://alex-snd.github.io/TRecover">
    <img src="https://img.shields.io/badge/docs-MkDocs-blue.svg?color=%2348466D" alt="MkDocs link"/>
  </a>
  <img src="https://img.shields.io/badge/python-v3.8.5-blue.svg?color=%2348466D" alt="Python version"/>
  <a href="https://colab.research.google.com/github/alex-snd/TRecover/blob/master/notebooks/TRecover-train-alone.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg?color=%2348466D" alt="Open In Colab"/>
  </a>
  <a href="https://badge.fury.io/py/trecover">
    <img src="https://img.shields.io/pypi/v/trecover?color=%2348466D" alt="PyPI version"/>
  </a>
  <img src="https://static.pepy.tech/personalized-badge/trecover?period=total&units=international_system&left_color=grey&right_color=%2348466D&left_text=pypi downloads" alt="PyPi Downloads"/>
  <a href="https://github.com/alex-snd/TRecover/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?color=%2348466D" alt="License Apache 2.0"/>
  </a>
</p>

## 🚀 Objective
The main goal of the project is to study the possibility of using Transformer neural network to “read” meaningful text in columns that can be compiled for a [Running Key Cipher](https://en.wikipedia.org/wiki/Running_key_cipher). You can read more about the problem [here](https://alex-snd.github.io/TRecover/objective/).

In addition, the second rather fun 😅 goal is to train a large enough model so that it can handle the case described below.
Let there be an original sentence:

>Hello, my name is ***Zendaya*** Maree Stoermer Coleman but you can just call me ***Zendaya***.

The columns for this sentence will be compiled in such a way that the last seven contain from ten to thirteen letters of the English alphabet, and all the others from two to five. Thus, the last seven characters will be much harder to "read" compared to the rest. However, we can guess from the meaning of the sentence that this is the name ***Zendaya***.
In other words, the goal is also to train a model that can understand and correctly “read” the last word.




## ⚙ Installation
Trecover requires Python 3.8 or higher and supports both Windows and Linux platforms.
1. Clone the repository:
```shell
git clone https://github.com/alex-snd/TRecover.git  && cd trecover
```

2. Create a virtual environment:
    * Windows:
    ```shell
    python -m venv venv
    ```
    * Linux:
    ```shell
    python3 -m venv venv
    ```
3. Activate the virtual environment:
    * Windows:
    ```shell
    venv\Scripts\activate.bat
    ```
    * Linux:
    ```shell
    source venv/bin/activate
    ```

5. Install the package inside this virtual environment:
    * Just to run the demo:
    ```shell
    pip install -e ".[demo]"
    ```
    * To train the Transformer:
    ```shell
    pip install -e ".[train]"
    ```
    * For development and training:
    ```shell
    pip install -e ".[dev]"
    ```
    
6. Initialize project's environment:
   ```shell
   trecover init
   ```
   For more options use:
   ```shell
   trecover init --help
   ```


## 👀 Demo
* 🤗 Hugging Face <br>
  You can play with a pre-trained model hosted [here](https://huggingface.co/spaces/alex-snd/TRecover).
* 🐳 Docker Compose<br>
  * Pull from Docker Hub:
    ```shell
    docker-compose -f docker/compose/scalable-service.yml up
    ```
  * Build from source:
    ```shell
    docker-compose -f docker/compose/scalable-service-build.yml up
    ```
* 💻 Local (requires docker) <br>
  * Download pretrained model:
    ```shell
    trecover download artifacts
    ```
  * Launch the service:
    ```shell
    trecover up
    ```



## 🗃️ Data
The [WikiText](https://huggingface.co/datasets/wikitext) and [WikiQA](https://huggingface.co/datasets/wiki_qa) datasets 
were used to train the model, from which all characters except English letters were removed.<br>
You can download the cleaned dataset:
```shell
trecover download data
```


## 💪 Train
To quickly start training the model, open the [Jupyter Notebook](https://colab.research.google.com/github/alex-snd/TRecover/blob/master/notebooks/TRecover-train-alone.ipynb).


* 🕸️ Distributed <br>
  TODO
* 💻 Local <br>
  After the dataset is loaded, you can start training the model:
  ```
  trecover train local \
  --project-name {project_name} \
  --exp-mark {exp_mark} \
  --train-dataset-size {train_dataset_size} \
  --val-dataset-size {val_dataset_size} \
  --vis-dataset-size {vis_dataset_size} \
  --test-dataset-size {test_dataset_size} \
  --batch-size {batch_size} \
  --n-workers {n_workers} \
  --min-noise {min_noise} \
  --max-noise {max_noise} \
  --lr {lr} \
  --n-epochs {n_epochs} \
  --epoch-seek {epoch_seek} \
  --accumulation-step {accumulation_step} \
  --penalty-coefficient {penalty_coefficient} \

  --pe-max-len {pe_max_len} \
  --n-layers {n_layers} \
  --d-model {d_model} \
  --n-heads {n_heads} \
  --d-ff {d_ff} \
  --dropout {dropout}
  ```
  For more information use `trecover train local --help`


## ✔️ Related work
TODO: what was done, tech stack.


## 🤝 Contributing
Contributions, issues and feature requests are welcome.<br />
Feel free to check [issues page](https://github.com/alex-snd/TRecover/issues) if you want to contribute.


## 👏 Show your support
Please don't hesitate to ⭐️ this repository if you find it cool!


## 📜 License
Copyright © 2022 [Alexander Shulga](https://www.linkedin.com/in/alex-snd).<br />
This project is [Apache 2.0](https://github.com/alex-snd/TRecover/blob/master/LICENSE) licensed.

