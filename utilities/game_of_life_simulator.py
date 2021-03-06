import matplotlib
import matplotlib.pyplot as plt
import seagull as sg
from seagull import lifeforms as lf
import preprocessing.rle_decoder as decoder

matplotlib.use('TkAgg')

board = sg.Board(size=(64, 64))
with open("../data-sources/non_oscillators/weekender.rle") as f:
    raw_content = f.read()
    raw_blinker = decoder.decode(raw_content)
    board.add(lifeform=lf.Custom(raw_blinker), loc=(30, 30))

sim = sg.Simulator(board)
run = sim.run(sg.rules.conway_classic, iters=1000)
animate = sim.animate()
plt.show()
