---
title: "🤷 About the project"
hide:

- navigation
- toc

---
<h1>     </h1>
<div align="center">
    <h2 align="center">Welcome to Text Recovery Project 👋 </h2>
    <h3 align="center">A python library for training a Transformer neural network to solve the <a href="https://en.wikipedia.org/wiki/Running_key_cipher">Running Key Cipher</a>, widely known in the field of cryptography.</h3>
    <img src="https://github.com/alex-snd/TRecover/blob/assets/preview_animation.gif?raw=true" alt="Preview Animation"/>
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
</div>

<h3 align="justify">
    The main goal of the project is to study the possibility of using Transformer neural network to “read” meaningful text 
    in columns that can be compiled for a <a href="https://en.wikipedia.org/wiki/Running_key_cipher">Running Key Cipher</a>. 
    You can read more about the problem <a href="https://alex-snd.github.io/TRecover/objective">here</a>.
    In addition, the second rather fun 😅 goal is to train a large enough model so that it can handle the case described
    below.<br><br>
    Let there be an original sentence:
</h3>

!!! attention ""

    <h3>Hello, my name is ***Zendaya*** Maree Stoermer Coleman but you can just call me ***Zendaya***.</h3>

<h3 align="justify">
    The columns for this sentence will be compiled in such a way that the last seven contain from ten to thirteen 
    letters of the English alphabet, and all the others from two to five. Thus, the last seven characters will be much 
    harder to "read" compared to the rest. However, we can guess from the meaning of the sentence that this is the 
    name <b><i>Zendaya</i></b>.
</h3>

!!! attention ""

    <h3>In other words, the goal is also to train a model that can understand and correctly “read” the last word.</h3>
