import tensorflow as tf
import sys
sys.path.append('../sf/package/')
import stealthflow as sf

print(dir(sf))

sf.layers.block.SEBlock(c=12)