---
title: "⚙️Installation"
---

# ⚙️Installation

Trecover requires Python 3.8 or higher and supports both Windows and Linux platforms.

## Installation steps:
### Clone the repository:

```shell
git clone https://github.com/alex-snd/TRecover.git  && cd trecover
```

### Create a virtual environment:
=== "Windows"
    ```PowerShell
    python -m venv venv
    ```
=== "Linux"
    ```Shell 
    python3 -m venv venv
    ```

### Activate the virtual environment:

=== "Windows"
    ```PowerShell
    venv\Scripts\activate.bat
    ```
=== "Linux"
    ```Shell
    source venv/bin/activate
    ```

### Install the package inside this virtual environment:

=== "Just to run the demo"
    === "From PyPi"
        ```shell
        pip install trecover[demo]
        ```
    === "From source"
        ```shell
        pip install -e ".[demo]"
        ```

=== "To train the Transformer"
    ```shell
    pip install -e ".[train]"
    ```

=== "To develop and train"
    ```shell
    pip install -e ".[dev]"
    ```



### Initialize project's environment:

```shell
trecover init
```
For more information use ``trecover --help`` or read the 
[reference](https://alex-snd.github.io/TRecover/src/trecover/app/cli/)

