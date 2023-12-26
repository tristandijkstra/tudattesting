from astroquery.mpc import MPC

a = (MPC.get_observations("1"))

print(a.colnames)

["number", "epoch", "RA", "DEC", "band", "observatory"]