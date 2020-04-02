# Venus on volcanoes

Download the data from Volcanoes on Venus Dataset: https://kdd.ics.uci.edu/databases/volcanoes/volcanoes.html and put it under data. Unpack and run data_loader.py.

## Install dependencies

```bash
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
```

## Keras / TF v2 issues with device selection

If there are device issues with tf v2 (experimental_device etc.) when using a CPU (or maybe even a GPU), go to

keras > backend > tensorflow_backend.py
 
and replace ```_get_available_gpus()``` with:

```python
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    global _LOCAL_DEVICES
    if _LOCAL_DEVICES is None:
        if _is_tf_1():
            devices = get_session().list_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
        else:
            devices = tf.config.list_logical_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in _LOCAL_DEVICES if 'device:gpu' in x.lower()]
``` 
