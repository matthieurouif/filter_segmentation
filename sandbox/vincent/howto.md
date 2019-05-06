# Matplotlib

If importing `matplotlib` in your Python code generates the following error:

```
RuntimeError: Python is not installed as a framework.
The Mac OS X backend will not be able to function correctly if Python is not installed 
as a framework. See the Python documentation for more information on installing Python 
as a framework on Mac OS X. Please either reinstall Python as a framework,
or try one of the other backends.
```

If you used `pip` to install `matplotlib`, here is a solution:

- Create the file `~/.matplotlib/matplotlibrc`
- Add the following line: `backend: TkAgg`

See https://matplotlib.org/users/customizing.html#the-matplotlibrc-file for more info


# Pip

You should make sure to use the correct `pip` to install Python packages in the Anaconda environment.
To see all the `pip` available :

```
which -a pip
```

When I'm in a conda environment, this gives me:

```
/anaconda3/envs/bg/bin/pip
/anaconda3/bin/pip
/usr/local/bin/pip
```

If I want to install a package in the `bg` environment, let's say matplotlib, I should use:

```
/anaconda3/envs/bg/bin/pip install matplotlib
```

# Python

What Python I'm using in the environment:

which python

List all Python:

which -a python

type python


# OpenCV


I know that OpenCV is install, but I have this anyway when I am in the Anaconda environment:

```
(bg) ➜  multiarray git:(master) ✗ python
Python 2.7.16 |Anaconda, Inc.| (default, Mar 14 2019, 16:24:02) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: No module named cv2
```

Solution:

```
(bg) ➜  multiarray git:(master) ✗ which -a pip
/anaconda3/envs/bg/bin/pip
/anaconda3/bin/pip
/usr/local/bin/pip
```

And

```
(bg) ➜  multiarray git:(master) ✗ /anaconda3/envs/bg/bin/pip install opencv-python
```
