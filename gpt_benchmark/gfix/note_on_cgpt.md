# Note on gauge fix function in cgpt

- [gauge_fix.cc](./gauge_fix.cc): the function `gauge_fix` in `cgpt`, for Coulomb gauge.
    - "err_on_no_converge" controls whether to throw an error if the gauge fixing does not converge;
    - "alpha" is the initial step size;

- [transform.py](./transform.py): the gpt function `gpt.gauge_fix` is defined, for Coulomb gauge.

Put [gauge_fix.cc](./gauge_fix.cc) in `lib/cgpt/lib/gauge_fix.cc`, put [transform.py](./transform.py) in `lib/gpt/core/transform.py`.

Then run `./make` in `lib/cgpt`, and modify `lib/gpt/core/__init__.py` to include the new function `gauge_fix`.