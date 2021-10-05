'''
This script is used for parsing the json schema for experiment settings.
'''
# %%
import sys
import inspect
import torch.nn as nn
import json
from collections import OrderedDict
import typing


# %%
def gen_object(type, default=None, recursive_item: OrderedDict = None) -> OrderedDict:
    _out = OrderedDict()
    if type == dict or type == OrderedDict:
        _out['description'] = None
        _out['type'] = 'object'
        _out['required'] = None
        _out['properties'] = OrderedDict()
        return _out
    elif type == int or type == float:
        _out['type'] = 'number'
    elif type == str:
        _out['type'] = 'string'
    elif type == bool:
        _out['type'] = 'boolean'
    elif type == list or type == tuple:
        _out['type'] = 'array'
        _out['items'] = recursive_item

    _out['default'] = default
    return _out


def post_process(schema: typing.Union[dict, OrderedDict, list]):
    if isinstance(schema, list):
        torm = []
        for i, v in enumerate(schema):
            if isinstance(v, (dict, list, OrderedDict)):
                schema[i] = post_process(v)
            elif v is None:
                torm.append(i)

        for i in torm[::-1]:
            schema.pop(i)
        return schema
    elif isinstance(schema, (dict, OrderedDict)):
        ptr_org = dict(schema)
        torm = []
        for k, v in ptr_org.items():
            if isinstance(v, (dict, list, OrderedDict)):
                ptr_org[k] = post_process(v)
            elif v is None:
                torm.append(k)
        for k in torm:
            del ptr_org[k]
        return ptr_org


def parse_processing(processing: typing.Union[dict, OrderedDict], module_list: list):
    processing['required'] = ['type', 'kwargs']
    properties = processing['properties']
    properties['type'] = gen_object(str)
    properties['type']['examples'] = [m.__name__ for m in module_list]

    # setup kwargs
    allOf = []
    for m in module_list:
        IfCondition = {'properties': {'type': {'const': m.__name__}}}
        ThenProcess = {'properties': {'kwargs': gen_object(dict)}}
        kwargs = ThenProcess['properties']['kwargs']
        allargspec = inspect.getfullargspec(m)
        args = [x for x in allargspec[0][::-1] if x != 'self']  # type:list
        defaults = allargspec[3]
        if defaults is None:
            defaults = []
        else:
            defaults = list(defaults[::-1])  # type:list
        defaults += [None for _ in range(len(args)-len(defaults))]
        anno = allargspec[-1]   # type:dict

        parsed = []
        for _arg, _default in zip(args, defaults):
            if _arg in anno:
                parsed.append((_arg, gen_object(anno[_arg], _default)))
            else:
                parsed.append((_arg, gen_object(None)))
                # raise RuntimeError(
                # f"{_arg} of {m.__name__} is not well annotated.")
        kwargs['properties'] = OrderedDict(parsed)

        allOf.append(OrderedDict([
            ('if', IfCondition),
            ('then', ThenProcess)
        ]))

    if len(module_list) == 1:
        processing['required'] = None
        processing['properties'] = allOf[0]['then']['properties']['kwargs']['properties']
    else:
        processing['allOf'] = allOf
    return processing

# %% [markdown]
# # Overall setting

# %%
schema = gen_object(dict)
schema['description'] = 'Settings of NN training.'
schema['required'] = ['transducer', 'encoder', 'decoder', 'joint', 'scheduler']

# %% [markdown]
# ## Transducer

# %%
from transducer_train import Transducer

processing = gen_object(dict)
parse_processing(processing, [Transducer])
processing['description'] = 'Configuration of Transducer'
schema['properties']['transducer'] = processing

# %% [markdown]
# ## Encoder

# %%
import model as am_zoo

processing = gen_object(dict)  # type:OrderedDict
modules = []
for m in dir(am_zoo):
    _m = getattr(am_zoo, m)
    if inspect.isclass(_m) and issubclass(_m, nn.Module):
        modules.append(_m)

parse_processing(processing, modules)
processing['description'] = 'Configuration of Transducer transcription network'
processing['properties']['freeze'] = gen_object(bool, default=False)
processing['properties']['pretrained'] = gen_object(bool, default=False)
schema['properties']['encoder'] = processing

# %% [markdown]
# ## Decoder

# %%
import lm_train as lm_zoo
processing = gen_object(dict)  # type:OrderedDict
modules = []
for m in dir(lm_zoo):
    if m in ['LSTMPredictNet', 'PlainPN']:
        _m = getattr(lm_zoo, m)
        modules.append(_m)

parse_processing(processing, modules)
processing['description'] = 'Configuration of Transducer prediction network'
processing['properties']['freeze'] = gen_object(bool, default=False)
processing['properties']['pretrained'] = gen_object(bool, default=False)
schema['properties']['decoder'] = processing

# %% [markdown]
# ## SpecAug

# %%
from _specaug import SpecAug

processing = gen_object(dict)  # type:OrderedDict
parse_processing(processing, [SpecAug])

processing['description'] = 'Configuration of SpecAugument'
schema['properties']['specaug_config'] = processing

# %% [markdown]
# ## Joint network

# %%
from transducer_train import JointNet, ConvJointNet


processing = gen_object(dict)  # type:OrderedDict
parse_processing(processing, [JointNet, ConvJointNet])

processing['description'] = 'Configuration of Transducer joint network'
schema['properties']['joint'] = processing

# %% [markdown]
# ## Scheduler

# %%
from torch.optim import Optimizer
import torch
import scheduler
from scheduler import Scheduler

processing = gen_object(dict)  # type:OrderedDict
modules = []
for m in dir(scheduler):
    _m = getattr(scheduler, m)
    if inspect.isclass(_m) and issubclass(_m, Scheduler):
        modules.append(_m)

parse_processing(processing, modules)
processing['description'] = 'Configuration of Scheduler'

# setup the optimizer
optim = gen_object(dict)
modules = []
for m in dir(torch.optim):
    _m = getattr(torch.optim, m)
    if inspect.isclass(_m) and issubclass(_m, Optimizer):
        modules.append(_m)
parse_processing(optim, modules)
optim['properties']['use_zero'] = gen_object(bool, default=True)
processing['properties']['optimizer'] = optim

schema['properties']['scheduler'] = processing

# %% [markdown]
# # Post processing and dump

# %%

schema = post_process(schema)
with open('.vscode/schemas.json', 'w') as fo:
    json.dump(schema, fo)


