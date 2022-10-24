---
title: "🤷 About the project"
hide:

- navigation
- toc

---
<h1 align="center"><img src="https://github.com/alex-snd/TRecover/blob/assets/preview_animation_white.gif?raw=true" alt="Preview Animation"/></h1>
<div align="center">
    <h2 align="center">Welcome to Text Recovery Project 👋 </h2>
    <h3 align="center">A python library for distributed training of a Transformer neural network across the Internet to solve the <a href="https://en.wikipedia.org/wiki/Running_key_cipher">Running Key Cipher</a>, widely known in the field of Cryptography.</h3>
    <a href="https://huggingface.co/spaces/alex-snd/TRecover">
        <img class="off-glb" src="https://img.shields.io/badge/demo-%F0%9F%A4%97%20Hugging%20Face-blue?color=%2348466D" alt="Hugging Face demo"/>
    </a>
    <a href="https://wandb.ai/snd/TRecover?workspace=user-snd">
        <img class="off-glb" src="https://img.shields.io/badge/visualize%20in-W&B-blue?color=%2348466D" alt="Visualize%20in%20W&B"/>
    </a>
    <a href="https://alex-snd.github.io/TRecover">
        <img class="off-glb" src="https://img.shields.io/badge/docs-MkDocs-blue.svg?color=%2348466D" alt="MkDocs link"/>
    </a>
    <img class="off-glb" src="https://img.shields.io/badge/python-v3.8.5-blue.svg?color=%2348466D" alt="Python version"/>
    <a href="https://badge.fury.io/py/trecover">
        <img class="off-glb" src="https://img.shields.io/pypi/v/trecover?color=%2348466D" alt="PyPI version"/>
    </a>
    <img class="off-glb" src="https://static.pepy.tech/personalized-badge/trecover?period=total&units=international_system&left_color=grey&right_color=%2348466D&left_text=pypi downloads" alt="PyPi Downloads"/>
    <a href="https://github.com/alex-snd/TRecover/blob/master/LICENSE">
        <img class="off-glb" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?color=%2348466D" alt="License Apache 2.0"/>
    </a>
</div>

<h3 align="justify">
    The main goal of the project is to study the possibility of using Transformer neural network to “read” meaningful text 
    in columns that can be compiled for a Running Key Cipher. 
    You can read more about the problem <a href="https://alex-snd.github.io/TRecover/objective/task_definition">here</a>.
    The second goal is to train a fairly large model in a distributed manner with the help of volunteers 
    from around the globe 🌎.
    <p>In addition, rather fun 😅 goal is to train a large enough model so that it can handle the case described
    below.<br>
    Let there be an original sentence:
    </p>
</h3>

!!! attention ""
    
    <font size="4">Hello, my name is ***Zendaya*** Maree Stoermer Coleman but you can just call me ***Zendaya***.</font>

<h3 align="justify">
    The columns for this sentence will be compiled in such a way that the last seven contain from ten to thirteen 
    letters of the English alphabet, and all the others from two to five. Thus, the last seven characters will be much 
    harder to "read" compared to the rest. However, we can guess from the meaning of the sentence that this is the 
    name <b><i>Zendaya</i></b>.
    <p>In other words, the goal is also to train a model that can understand and correctly “read” the last word.</p>
</h3>


