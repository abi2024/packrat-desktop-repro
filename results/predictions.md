# Thin (8x1) vs fat (1x8). BERT eager, 16-token input. Written 2026-11-02, before the run.
Batch 8: fat wins or ties. Thin's best case (127 ms) already equals fat (131 ms), so any interference makes thin lose.
Batch 32: close. Thin lands between 420 and 500 ms (fat 473, ideal ~367). I think fat should do well here. Memory is how fast the weights reach the cores, and that is central to thin: each thin instance reads the full weights for only 4 short sentences, so 8 instances hit RAM 8x as often, and at 2133 MT/s that eats most of thin's headroom.
No call on 2x4 or 4x2; I only considered the two extremes.
