# Install guide

## Conda

Download and install Anaconda ([see here](https://www.anaconda.com/)).

Let's see if the `conda` command is available. Open a terminal and try to execute the command:

```
> conda
```

If `conda` command is not available, add conda to your PATH by executing the following command in a terminal:

```
> export PATH=/anaconda3/bin/:$PATH
```

The `conda` command should be available now in this terminal and as long as you don't close it.

Even better, add this line in your .bashrc or .zshrc file so that the `conda` command will be available every time a terminal is opened.


## Create virtual environment

Use the following command line to create the environment nammed `frida` (any name will do):

```
> conda create -n frida python=3.6
```

If you want to see all the available environments in conda:

```
> conda env list
```

or 

```
> conda info --envs
```


## Access the virtual environment

Access the environment using:

```
> source activate frida
```

Close the environment using:

```
> source deactivate
```

## Install Python packages

First, let's check what's install using:

```
> pip freeze
```

This gave me the following:

```
certifi==2019.3.9
```

Let's install the necessary package:

```
> pip numpy matplotlib coremltools
```

## Start Jupyter notebook

- Make sure you are in the `frida` environment.
- In the terminal, go to the folder where you want to open/create a notebook.
- Execute:

```
> jupyter notebook
```

The notebook should open itself in a webpage.