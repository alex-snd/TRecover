---
title: "🚀 How to join training"
---

<h1 align="left">Join Collaborative Training</h1>

!!! failure "Collaborative training hasn't started yet"

## 😎 Easy way

<p align="justify">
<font size="4">
You can help to train the model by running a pre-prepared Jupyter Notebook on Kaggle or Google Colab.
To join the collaborative training, all you have to do is to keep the Notebook running for 
<a><b>at least 15 minutes</b></a> (but more is better) and you're free to close it 
after that and join again in another time.
</font>
</p>

### <font size="5">Join as a client peer</font>

<p align="justify">
<font size="4">
Kaggle gives you around 40 hrs per week of GPU time, so it's preferred over Colab, unless you have
Colab Pro or Colab Pro+.
</font>
</p>

=== "<font size="5">:kaggle:</font><font size="4">Kaggle (recommended)</font>"
    <img src="https://github.com/alex-snd/TRecover/blob/assets/kaggle_animation.gif?raw=true"/>

    ??? success "Please make sure to select GPU accelerator and switch the "Internet" ON, in kernel settings"
        <font size="3">If you are a new Kaggle member, you will need to verify your phone number in the 
        settings pane to turn on the Internet.
        </font>
        <img src="https://github.com/alex-snd/TRecover/blob/assets/kaggle_ensure.png?raw=true"/>

    !!! danger "Warning: please don't use multiple Kaggle accounts at once"

    <div align="center">[Open in Kaggle](https://www.kaggle.com/code/alexsnd/collaborativetrain){ .md-button }</div>

=== "<font size="5">:colab:</font><font size="4">Google Colab</font>"
    <img src="https://github.com/alex-snd/TRecover/blob/assets/collab_animation.gif?raw=true"/>

    ??? success "Please make sure to select GPU accelerator"
        <img src="https://github.com/alex-snd/TRecover/blob/assets/colab_ensure.png?raw=true"/>

    !!! danger "Warning: please don't use multiple Google Colab accounts at once"

    <div align="center">[Open in Colab](https://colab.research.google.com/github/alex-snd/TRecover/blob/master/notebooks/TRecover-train-colab.ipynb){ .md-button }</div>




## 😊 Preferred way

<p align="justify">
<font size="4">
You can join the collaborative training as an auxiliary peer if you have a Linux computer with 15+ GB RAM, at least 
100 Mbit/s download&upload speed and one port opened to incoming connections. Also, if in addition to all this there is also 
a GPU, then you can join as a trainer peer.
</font>
</p>

??? question "Why is this way more preferable?"
    <p align="justify">
    <font size="3">
    There are two broad types of peers: normal (full) peers and client mode peers. Client peers rely on others to 
    average their gradients, but otherwise behave the same as full peers. This way of participation is preferable as 
    the auxiliary and trainer peers not only don’t rely on others, but also can serve as relays and help others with 
    all-reduce.    
    </font>
    </p>

### <font size="5">Installation</font>

<p align="justify">
<font size="4">
This step is common to the trainer and auxiliary peer, and requires Python 3.8 or higher.
Open a <a href="https://tmuxcheatsheet.com">tmux</a> (or screen) session that will stay up after you logout and follow 
the instructions below.
</font>
</p>

<div class="termy" align="center">
```console
// Clone the repository
$ git clone https://github.com/alex-snd/TRecover.git
Cloning into 'TRecover'...    
remote: Enumerating objects: 4716, done.
remote: Counting objects: 100% (368/368), done.
remote: Compressing objects: 100% (133/133), done.
remote: Total 4716 (delta 196), reused 348 (delta 181)
---> 100%
Receiving objects: 100% (4716/4716), 18.69 MiB | 4.36 MiB/s, done.
Resolving deltas: 100% (2908/2908), done.
<br>
// Change the working dir
$ cd TRecover
<br>
// Create a virtual environment
$ python3 -m venv venv
<br>
// Activate the virtual environment
$ source venv/bin/activate
<br>
// Install the package
$ pip install -e .[collab]
<b>Successfully installed trecover</b>
<br>
// Initialize project's environment
$ trecover init
<b>Project's environment is initialized.</b>
<br>
<br>
// For more information use <font color="#36464E">trecover --help</font> or read the <a href="https://alex-snd.github.io/TRecover/src/trecover/app/cli">reference</a>.
```
</div>


### <font size="5">Join as a trainer peer</font>

<p align="justify">
<font size="4">
Trainers are peers with GPUs (or other compute accelerators) that compute gradients, average them via all-reduce and 
perform optimizer steps. 
</font>
</p>

<div class="termy" align="center">

    ```console
    // Download the train dataset
    $ trecover download data
    ---> 100%
    Downloaded "data.zip" to ../data/data.zip
    Archive extracted to ../data

    // Run the trainer peer
    $ trecover collab train --sync-args --batch-size 1 --n-workers 2 --backup-every-step 1
    
    // Use <font color="#36464E">--help</font> flag for more details
    ```

</div>


### <font size="5">Join as an auxiliary peer</font>

<p align="justify">
<font size="4">
Auxiliary peers are low-end servers without GPU that will keep track of the latest model checkpoint 
and assist in gradient averaging. 
</font>
</p>

<div class="termy" align="center">

    ```console
    // Run the auxiliary peer
    $ trecover collab aux --sync-args --verbose --as-active-peer --backup-every-step 1
    
    // Use <font color="#36464E">--help</font> flag for more details
    ```

</div>