# `known_shift_real.mp4`

A real H.264-encoded video (produced by a real writer -- `ffmpeg`/`libx264`,
not a synthetic byte layout this codebase constructed itself), used to
validate `pinneapple_perception.video_piv.piv_velocity_field`/
`piv_velocity_sequence` against an actual video encode/decode round-trip
(compression, chroma subsampling, codec block artifacts) rather than only
the pure-numpy-array synthetic cases in `tests/test_perception.py`.

## Provenance (exact reproduction steps)

1. A 170x170 random-noise texture was generated with numpy
   (`np.random.seed(7)`).
2. 6 frames were rendered by shifting that texture with
   `scipy.ndimage.shift(base, shift=(vy*t, vx*t), order=3, mode='wrap')`
   for `t = 0..5`, with a known constant sub-pixel velocity
   `vx=2.3, vy=-1.7` (pixels/frame), then cropped to 128x128 and saved as
   PGM.
3. Encoded into this file with:
   `ffmpeg -framerate 10 -i f%03d.pgm -c:v libx264 -crf 18 -pix_fmt yuv420p known_shift_real.mp4`

This means the file is a genuine libx264-compressed video, not a
hand-constructed byte layout -- reading it back requires a real video
decoder (`ffmpeg`, used by the test via subprocess) to get raw frames, at
which point the frames carry real compression artifacts absent from the
original numpy arrays.

## Expected result

`tests/test_perception.py::test_piv_on_real_encoded_video_recovers_known_shift`
decodes this file with `ffmpeg` and asserts the recovered PIV velocity
matches the known `(vx, vy) = (2.3, -1.7)` px/frame to within the same
~0.3px tolerance established for the synthetic sub-pixel test (the
well-documented cross-correlation "peak-locking" bias, not a defect) --
confirmed empirically before being written into the test: mean
u=2.21±0.07, v=-1.79±0.06 across all 5 frame pairs.
