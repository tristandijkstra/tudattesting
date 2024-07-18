sat_ephemeris = {}

satellites_Horizons_codes = ["C57"]

for code in satellites_Horizons_codes:
    temp = HorizonsQuery(
        query_id=code,
        location=f"@{global_frame_origin}",
        epoch_start=epoch_start_buffer,
        epoch_end=epoch_end_buffer,
        epoch_step=f"{int(timestep_global/60)}m",
        extended_query=True,
    )

    sat_ephemeris[code] = temp.create_ephemeris_tabulated(
        frame_origin=global_frame_origin,
        frame_orientation=global_frame_orientation,
    )


# List the bodies for our environment
bodies_to_create = [
    "Sun",
]

# Create system of bodies
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Add satellite ephemerides
for name in satellites_names:
    body_settings.add_empty_settings(name)
    body_settings.get(name).ephemeris_settings = sat_ephemeris[name]


bodies = environment_setup.create_system_of_bodies(body_settings)
