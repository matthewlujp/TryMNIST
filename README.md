# TryMNIST
__Try mnis with Tensorflow__

Introduce Tensorflow
---
See [official documnet](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation)  
*This introduction is for Python2 on mac.*  

1. Introduce virtualenv for Python.
  ```bash
  $ sudo easy_install pip
  $ sudo pip install virtualenv
  ```
  
2. Create a virtual environment for to which Tensorflow is installed
  ```bash
  # Somewhere you want to create tensorflow directory
  $ mkdir tensorflow
  $ virtaulenv tensorflow
  ...
  ```
  Start that virtual environment
  ```bash
  $ cd tensorflow
  $ source bin/activate
  ```
  
3. Install Tensorflow
  ```bash
  # only cpu version for mac
  $ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl
  ```
  gpu enabled version is not available for mac. Check [this](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation) out.
  
4. Try out Tensorflow
  Try following.
  ```python
  import tensorflow as tf

  hello = tf.constant('hello tensorflow!')
  print(hello.eval())

  session = tf.InteractiveSession()
  a = tf.constant(10)
  b = tf.constant(20)
  print(session.run(a + b))
  ```
  
5. Clone this repostory and run mnist_convolution.py
  ```bash
  python mnist_convolution.py
  ```
  It will take so long time.
  If you either increase or reduce learning steps, tweak ```def learn()``` a bit.
  ```python
  for i in range(1000):
    batch = mnist.train.next_batch(50)
    ...
   ```
   
  
__Congrats!__
