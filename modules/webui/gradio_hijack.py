from gradio.components.base import Block

all_components = []

if not hasattr(Block, "original__init__"):
    Block.original_init = Block.__init__


def blk_ini(self, *args, **kwargs):
    all_components.append(self)
    return Block.original_init(self, *args, **kwargs)


Block.__init__ = blk_ini
