# Thin (8x1) vs fat (1x8). BERT eager, 16-token input. Written 2026-11-02, before the run.
Batch 8: fat wins or ties. Thin's best case (127 ms) already equals fat (131 ms), so any interference makes thin lose.
Batch 32: close. Thin lands between 420 and 500 ms (fat 473, ideal ~367). Because each thin instance reads all the weights for only 4 short sentences, so 8 instances hit RAM 8x as often, and at 2133 MT/s that penalty eats most of thin's headroom.
