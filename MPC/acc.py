from tudatpy.kernel.numerical_simulation import propagation_setup

bods = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Moon",
    "Mars",
    # "Phobos",
    # "Deimos",
    "Jupiter",
    # "Ganymede",
    # "Callisto",
    # "Io",
    # "Europa",
    # "Amalthea",
    "Saturn",
    # "Titan",
    # "Rhea",
    # "Iapetus",
    # "Dione",
    # "Tethys",
    # "Enceladus",
    # "Mimas",
    # "Hyperion",
    "Uranus",
    "Neptune",
]

allAccels = {
    "Sun": [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.relativistic_correction(True),
        # propagation_setup.acceleration.cannonball_radiation_pressure(),
    ],
    # Mercury system
    "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
    # Venus system
    "Venus": [propagation_setup.acceleration.point_mass_gravity()],
    # Earth system
    "Earth": [propagation_setup.acceleration.point_mass_gravity()],
    "Moon": [propagation_setup.acceleration.point_mass_gravity()],
    # Mars system
    "Mars": [propagation_setup.acceleration.point_mass_gravity()],
    # "Phobos": [propagation_setup.acceleration.point_mass_gravity()],
    # "Deimos": [propagation_setup.acceleration.point_mass_gravity()],
    # Asteroid belt
    # "Ceres": [propagation_setup.acceleration.point_mass_gravity()],
    # Jupiter system
    "Jupiter": [propagation_setup.acceleration.point_mass_gravity()],
    # "Ganymede": [propagation_setup.acceleration.point_mass_gravity()],
    # "Callisto": [propagation_setup.acceleration.point_mass_gravity()],
    # "Io": [propagation_setup.acceleration.point_mass_gravity()],
    # "Europa": [propagation_setup.acceleration.point_mass_gravity()],
    # "Amalthea": [propagation_setup.acceleration.point_mass_gravity()],
    # Saturn system
    "Saturn": [propagation_setup.acceleration.point_mass_gravity()],
    # "Titan": [propagation_setup.acceleration.point_mass_gravity()],
    # "Rhea": [propagation_setup.acceleration.point_mass_gravity()],
    # "Iapetus": [propagation_setup.acceleration.point_mass_gravity()],
    # "Dione": [propagation_setup.acceleration.point_mass_gravity()],
    # "Tethys": [propagation_setup.acceleration.point_mass_gravity()],
    # "Enceladus": [propagation_setup.acceleration.point_mass_gravity()],
    # "Mimas": [propagation_setup.acceleration.point_mass_gravity()],
    # "Hyperion": [propagation_setup.acceleration.point_mass_gravity()],
    # "Phoebe": [propagation_setup.acceleration.point_mass_gravity()],
    # Uranus system
    "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
    # "Titania": [propagation_setup.acceleration.point_mass_gravity()],
    # "Oberon": [propagation_setup.acceleration.point_mass_gravity()],
    # "Ariel": [propagation_setup.acceleration.point_mass_gravity()],
    # "Umbriel": [propagation_setup.acceleration.point_mass_gravity()],
    # Neptune system
    "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
    # "Triton": [propagation_setup.acceleration.point_mass_gravity()],
    # Neptune system
}
