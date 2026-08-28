include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  map_frame = "map",
  tracking_frame = "base_scan",
  published_frame = "base_link",
  odom_frame = "odom",

  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,

  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,

  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,

  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 1.0,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,

  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

MAP_BUILDER.use_trajectory_builder_2d = true

TRAJECTORY_BUILDER_2D.min_range = 0.2
TRAJECTORY_BUILDER_2D.max_range = 6.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.0

TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1

TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)

TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 2,
}

POSE_GRAPH.optimize_every_n_nodes = 20

-- Tightened 2026-08-27 after the pose was caught flip-flopping between two
-- hypotheses ~4-5m apart mid-drive (origin vs (-4.1,-1.0)), inside a SINGLE
-- cartographer session -- i.e. false constraint matches winning, not a
-- restart. Two stretches of this track look alike to the LiDAR, and with no
-- motion prior at all (use_odometry=false, use_imu_data=false above) scan
-- matching is weakest exactly on hard curves, which is where it teleported.
--
-- min_score gates LOCAL constraints against the frozen map -- the primary
-- localization mechanism in pure_localization, so this one is raised only
-- modestly. global_localization_min_score gates full-window relocalization,
-- which is the fallback that was firing wrongly, so it is raised hard.
POSE_GRAPH.constraint_builder.min_score = 0.78
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.90

return options