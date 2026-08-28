# QCar Physical Deployment — Progress Log

**Last updated:** 2026-08-20
**Status:** Full pipeline working end-to-end — the car drives itself autonomously around the recorded track, with LiDAR-based autonomous overtaking, depth-camera emergency stop, and an obstacle voice announcement all confirmed working on physical hardware.

For full technical detail (pipeline architecture, node-by-node breakdown, topic map, exact commands, debugging history) see **`notes.md`**. This file is just the progress log.

---

## Progress

```
✅ Physical hardware driving confirmed
✅ Mapping (Cartographer) confirmed
✅ Localization confirmed (switched from AMCL to Cartographer — AMCL drifted
     over a full lap, Cartographer's pose-graph optimization doesn't)
✅ Trajectory recording — clean closed lap (2.1cm start/end gap)
✅ Trajectory processing — smoothed into final .npy reference trajectory
✅ MPC controller — car drives the recorded path fully autonomously,
     with LiDAR-based obstacle avoidance and depth-camera emergency stop
✅ Obstacle voice announcement ("Obstacle detected", English) on a full
     stop — not on a normal successful overtake
✅ Autonomous overtaking confirmed working on physical hardware — car
     detects an obstacle on a straight, swings out, passes it, and
     returns to lane, tested over multiple 3-lap runs. Fixed several
     real bugs to get here: overtake distance thresholds that made
     overtaking either lose to emergency-stop or never trigger at all,
     and a loop-closure bug that could leave the car "stuck" (crawling
     straight instead of returning to lane) if an overtake happened
     near the trajectory's start/end seam. Full detail in notes.md,
     Issues 14-15.
      │
      ▼
🔲 Capture the roundabout in the recorded trajectory (missed in every
     recording attempt so far — not a tooling problem, just need to
     explicitly drive it next time)
🔲 Try lane centering (camera-based) alongside MPC path following —
     fully implemented and pre-configured, just never tested
🔲 Overtake reaction could still be tuned further (distance/timing) —
     works reliably now, not yet fully polished
🔲 `/motion_enable false` still doesn't reliably stop the car mid-drive
     even after fixing the topic-collision bug — root cause still open
```

## What's Been Done (short version)

Got the whole pipeline working on the physical car: mapped the room with Cartographer, got localization solid (switched from AMCL to Cartographer partway through after AMCL turned out to drift on this track), recorded and processed a clean closed-loop trajectory, and got the MPC controller driving it fully autonomously with LiDAR + depth-camera safety systems active. Along the way, fixed a string of real bugs — spline oversmoothing cutting corners, a depth-sensor emergency-stop that could get permanently stuck, localization drift that got worse lap over lap, a topic collision that broke manual stop commands, and a missing audio-mixer setting that meant the obstacle announcement was inaudible.

Since then, got autonomous overtaking working reliably: fixed the overtake-vs-emergency-stop distance thresholds, found and fixed a hidden default in the LiDAR sector analyzer that was silently capping obstacle detection at 0.75m regardless of any other setting (making overtaking impossible once the "far enough to overtake" floor got raised above it), and fixed a loop-closure tolerance bug that could leave the car unable to return to lane after overtaking near the trajectory's loop seam.

Everything used to reproduce this — full command sequences, why each fix was needed, and how the nodes actually talk to each other — is in `notes.md`.
