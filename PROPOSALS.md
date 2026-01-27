# Gaze Tracking Improvement Proposals

This document outlines proposed improvements and how they are expected to impact accuracy.

## Core Accuracy Improvements

1) Head-pose compensation
- What: Estimate head pose (yaw/pitch/roll) from FaceMesh and compensate gaze features before mapping.
- Expected impact: Reduced drift and systematic offsets when users move their head.

2) Iris-corner normalized features
- What: Compute gaze features from iris center relative to eye corner landmarks, instead of raw ratios.
- Expected impact: More stable features across distance and lighting changes; improved cross-session consistency.

3) Outlier rejection during calibration
- What: Discard frames with blink or low-confidence/unstable landmarks during calibration.
- Expected impact: Cleaner calibration fit, lower mean error in test stage.

## Stability and Smoothing

4) One-Euro or Kalman filtering
- What: Replace cluster heuristic with a temporal filter on (x, y).
- Expected impact: Lower jitter and better perceived accuracy; tune to avoid latency.

5) Reset smoothing on face loss
- What: If face is lost, return (None, None) and reset filter state.
- Expected impact: Prevents stale gaze outputs that look inaccurate.

## Calibration UX and Quality

6) Calibration metadata + validation
- What: Save screen size, timestamp, FPS, model version in calibration files and validate on load.
- Expected impact: Prevents reusing invalid calibrations that cause large systematic error.

7) Quality score after calibration
- What: Report mean/median error and valid-frame ratio after test stage.
- Expected impact: Quick signal when calibration is unreliable.

8) Single-point re-calibration
- What: Allow recalibrating a single point rather than repeating full calibration.
- Expected impact: Fixes local errors without full rerun; improves overall fit.

## Reliability and Robustness

9) Clear landmarks on no-face detection
- What: Ensure stale landmarks are not reused when detection fails.
- Expected impact: Avoids incorrect gaze outputs that skew accuracy.

10) Tighten webapp locking + cleanup on disconnect
- What: Release camera/API resources if a client disconnects mid-session.
- Expected impact: Prevents hung state and inconsistent calibration data.

## Project Structure

11) Split dependencies
- What: Separate requirements for web vs CV features (requirements-web.txt, requirements-cv.txt).
- Expected impact: Cleaner installs; fewer dependency conflicts.

12) Move outputs into a dedicated runs/ directory
- What: Store logs and data artifacts outside tracked source tree.
- Expected impact: Cleaner repo, fewer accidental changes committed.
