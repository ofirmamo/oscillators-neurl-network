import matplotlib.pyplot as plt
import seagull as sg
import os
from seagull import lifeforms as lf

from lifeforms_gen import rle_decoder as decoder

board = sg.Board(size=(64, 64))
with open("../data-sources/non_oscillators/weekender.rle") as f:
    raw_content = f.read()
    raw_blinker = decoder.decode(raw_content)
    board.add(lifeform=lf.Custom(raw_blinker), loc=(30, 30))

sim = sg.Simulator(board)
stats = sim.run(sg.rules.conway_classic, iters=1000)
anim = sim.animate()
plt.show()
