def reward_function(params):
    # Example of penalize steering, which helps mitigate zig-zag behaviors

    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    abs_steering = abs(params['steering_angle']) # Only need the absolute steering angle
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']

    SPEED_THRESHOLD = 1.0
    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 15

    # Calculate 3 marks that are farther and father away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.35 * track_width
    marker_3 = 0.5 * track_width

    # Penalize reward heavily if car is out of track
    if (distance_from_center > marker_3) or (not all_wheels_on_track):
        reward = 1e-3

    # Penalize reward if the car is steering too much
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8

    if all_wheels_on_track and distance_from_center <= marker_2 and speed > SPEED_THRESHOLD:
        reward = 1.0
    elif all_wheels_on_track and distance_from_center > marker_2 and distance_from_center <= marker_3 and speed <= SPEED_THRESHOLD:
        reward = 0.3
    elif all_wheels_on_track and distance_from_center > marker_2 and distance_from_center <= marker_3 and speed > SPEED_THRESHOLD:
        reward = 1.0
    # likely crashed/ close to off track
    else:
        reward = 1e-3

        
    return float(reward)