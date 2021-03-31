# alice

Oblate fluid planet structure from a fourth-order theory of figures based on [Nettelmann 2017](https://doi.org/10.1051/0004-6361/201731550).

## Getting started

Set up by setting these environment variables or adding to your shell startup script:
```
export ALICE_DIR=~/alice # or whatever path you've cloned the repo to
export PYTHONPATH=$ALICE_DIR:$PYTHONPATH
```
(or equivalent for your shell).

Then unpack the eos data:
```
tar xvfz eos_data.tar.gz
```
and you're ready test the code by running
```
python gravity.py
```
to relax an oblate Saturn model. If all goes well this test model will converge and save the underlying model to `output/{uid}/tof4_data.pkl` where `{uid}` is an integer label assigned to the model when it is created.