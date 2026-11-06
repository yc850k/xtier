"""A split-aware GGUF reader: the checkpoint's shards read as one file.

llama.cpp's split checkpoints (``model-00001-of-00004.gguf`` and its parts) keep the metadata in the first part
and the tensors in the parts. The ``gguf`` package's reader opens one file, so SGLang's GGUF loader sees only the
first part's tensors (42 of DeepSeek V4.1's 1008). This reader merges the parts' tensor tables: the metadata and the
tensor count of the first part, the tensors of all the parts (each part's own reader, so the bytes are the part's).

The plugin installs it (see ``install``): SGLang's loader calls ``gguf.GGUFReader(path)`` and reads the tensors.
"""

import glob
import logging
import os
import re

logger = logging.getLogger(__name__)

_PART = re.compile(r'^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$')


def parts(path):
    """The checkpoint's parts (the given file first): the files named ``stem-00001-of-0000N.gguf``."""
    path = str(path)
    m = _PART.match(os.path.basename(path))
    if not m:
        return [path]
    found = sorted(glob.glob(os.path.join(os.path.dirname(path), f'{m.group("stem")}-*-of-*.gguf')))
    if not found:
        return [path]
    return found


class SplitReader:
    """The tensors of a split GGUF: the metadata of the first part, the tensor tables of all the parts."""

    def __init__(self, path, mode='r', reader=None):
        import gguf

        reader = reader or gguf.GGUFReader
        self.paths = parts(path)
        self.readers = [reader(p, mode) for p in self.paths]
        first = self.readers[0]
        self.fields = first.fields
        self.tensors = []
        for reader in self.readers:
            self.tensors.extend(reader.tensors)
        self.data = first.data
        self.data_offset = first.data_offset
        self.alignment = first.alignment
        self.byte_order = first.byte_order
        self.endianess = first.endianess
        if len(self.readers) > 1:
            logger.info('xtier: the GGUF checkpoint is split into %d parts (%s): %d tensors',
                        len(self.readers), os.path.basename(self.paths[0]), len(self.tensors))

    def get_field(self, key):
        return self.fields.get(key, None)

    def get_tensor(self, idx):
        return self.tensors[idx]


def install():
    """Make the gguf package's reader read a split checkpoint (the parts merged).

    The patch is on the module's attribute, so every later import of it (SGLang's loader imports gguf inside the
    function) sees the split reader.
    """
    import gguf

    if getattr(gguf.GGUFReader, '_xtier_split', False):
        return False
    original = gguf.GGUFReader

    def reader(path, mode='r'):
        found = parts(path)
        if len(found) <= 1:
            return original(path, mode)
        return SplitReader(path, mode, reader=original)

    reader._xtier_split = True
    reader._xtier_original = original
    gguf.GGUFReader = reader
    logger.info('xtier: the GGUF reader reads split checkpoints (the parts merged)')
    return True


def install_loader():
    """SGLang's GGUF loader takes the checkpoint's file; a directory (the model's, with its config) is resolved to
    the split's first part."""
    try:
        from sglang.srt.model_loader.loader import GGUFModelLoader
    except ImportError:
        return False
    if getattr(GGUFModelLoader, '_xtier_split', False):
        return False
    original = GGUFModelLoader._prepare_weights

    def prepare_weights(self, model_name_or_path):
        path = str(model_name_or_path)
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, '*.gguf')))
            if not found:
                raise ValueError(f'xtier: {path} has no GGUF file')
            path = found[0]
            logger.info('xtier: the GGUF checkpoint of %s: %s', path, os.path.basename(path))
        return original(self, path)

    GGUFModelLoader._prepare_weights = prepare_weights
    GGUFModelLoader._xtier_split = True
    logger.info('xtier: the GGUF loader takes a model directory (the split\'s first part)')
    return True
