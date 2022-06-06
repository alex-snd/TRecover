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




## 👀 Demo
demo (huggingface gif), docker🐳 online, docker local (pull, build),


## ⚙ Installation
trecover up command


## 🗃️ Data


## 💪 Train
About data (download, create custom), colab badge
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-snd/TRecover/blob/master/notebooks/TRecover-train-alone.ipynb)


## ✔️ Related work
what was done, tech stack


## 🤝 Contributing
Contributions, issues and feature requests are welcome.<br />
Feel free to check [issues page](https://github.com/alex-snd/TRecover/issues) if you want to contribute.


## 👏 Show your support
Please don't hesitate to ⭐️ this repository if you find it cool!


## 📜 License
Copyright © 2022 [Alexander Shulga](https://www.linkedin.com/in/alex-snd).<br />
This project is [Apache 2.0](https://github.com/alex-snd/TRecover/blob/master/LICENSE) licensed.

