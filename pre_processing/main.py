import matplotlib
import matplotlib.pyplot as plt
import seagull as sg
from seagull import lifeforms as lf

from lifeforms_gen import rle_decoder as decoder

matplotlib.use('TkAgg')

board = sg.Board(size=(64, 64))
with open("../data-sources/oscillators/38p7.2.rle") as f:
    raw_content = f.read()
    raw_blinker = decoder.decode(raw_content)
    board.add(lifeform=lf.Custom(raw_blinker), loc=(30, 30))

sim = sg.Simulator(board)
stats = sim.run(sg.rules.conway_classic, iters=1000)
anim = sim.animate()
plt.show()


board = sg.Board(size=(64, 64))
with open("../data-sources/oscillators/eureka.rle") as f:
    raw_content = f.read()
    raw_blinker = decoder.decode(raw_content)
    board.add(lifeform=lf.Custom(raw_blinker), loc=(30, 30))

sim = sg.Simulator(board)
stats = sim.run(sg.rules.conway_classic, iters=1000)
anim = sim.animate()
plt.show()