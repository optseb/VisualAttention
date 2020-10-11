You need to tell conda to install the same cuda toolkit as the one you
have installed on your system. I'm using Ubuntu 18.04 and its own apt
managed cuda toolkit, which is version 9.

So:

```
sudo apt install nvidia-cuda-toolkit
conda install cudatoolkit=9.0
```

That gives me a working system.
