---
title: "⚙️Installation"
---

# ⚙️Installation

Trecover requires Python 3.8 or higher and supports both Linux and Windows (restricted) platforms.

## Installation steps

=== "Linux"
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
    // Install the package (see installation options below)
    $ pip install -e .
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
=== "Windows"
    <div class="termy" align="center">
    ```PowerShell
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
    $ python -m venv venv
    <br>
    // Activate the virtual environment
    $ venv\Scripts\activate.bat
    <br>
    // Install the package (see installation options below)
    $ pip install -e .
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


## Installation options

=== "Just to run the demo"
    === "From PyPi"
        ```shell
        pip install trecover[demo]
        ```
    === "From source"
        ```shell
        pip install -e ".[demo]"
        ```

=== "To train"
    ```shell
    pip install -e ".[train]"
    ```

=== "To develop"
    ```shell
    pip install -e ".[dev]"
    ```