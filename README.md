# alice
Fluid planet interiors with ToF4. 

Depends on [ongp](https://github.com/chkvch/ongp) to make a spherical initial guess model and to provide interfaces to equations of state.

### Usage
See the main program in `gravity.py` for example usage.

### Output
For a converged model `gravity.tof4.relax()` will write the final model (in the form of a pickled python dictionary) to `output/{uid}/tof4_data.pkl` where `{uid}` is sort-of-unique identifying integer (Unix time in microseconds when model is intialized). This can be loaded with something like
```
import pickle
tof = pickle.load(open('output/1597971138122883/tof4_data.pkl', 'rb'))

print(list(tof)) # ['params', 'mesh_params', 'scalars', 'vectors']
print(tof['vectors']['rho']) # [[6.96938825e+00 6.96824039e+00 6.96479720e+00 ... ]

```