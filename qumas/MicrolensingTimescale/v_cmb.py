import numpy as np

def ra_dec_to_unit_vector(ra, dec):
    """
    Convert RA and Dec to a unit vector.
    
    Parameters:
    ra  -- Right Ascension in degrees
    dec -- Declination in degrees
    
    Returns:
    A 3-element numpy array representing the unit vector.
    """
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    return np.array([x, y, z])

def calculate_projections(ra0, dec0,cmb_ra=167.9,cmb_dec = -6.9):
    """
    Calculate the projections of the CMB dipole on East and North directions.
    
    Parameters:
    cmb_ra  -- RA of the CMB dipole in degrees (J2000)
    cmb_dec -- Dec of the CMB dipole in degrees (J2000)
    ra0     -- RA of the point of interest in degrees (J2000)
    dec0    -- Dec of the point of interest in degrees (J2000)
    
    Returns:
    A tuple with (v_north, v_east), the projections of the dipole on the North and East directions.
    """
    # Convert CMB dipole direction to unit vector
    v_cmb = ra_dec_to_unit_vector(cmb_ra, cmb_dec)
    
    # Calculate the unit vector for North direction
    dec0_rad = np.radians(dec0)
    ra0_rad = np.radians(ra0)
    
    v_north = np.array([-np.sin(dec0_rad) * np.cos(ra0_rad),
                        -np.sin(dec0_rad) * np.sin(ra0_rad),
                         np.cos(dec0_rad)])
    
    # Calculate the unit vector for East direction
    v_east = np.array([-np.sin(ra0_rad),
                        np.cos(ra0_rad),
                        0])
    
    # Project the CMB dipole onto the North and East directions
    projection_north = np.dot(v_cmb, v_north)
    projection_east = np.dot(v_cmb, v_east)
    
    return projection_north, projection_east

# # Example usage:
# cmb_ra = 167.9  # CMB dipole RA in degrees (example value)
# cmb_dec = -6.9  # CMB dipole Dec in degrees (example value)
# ra0 = 30.0      # RA of the point of interest in degrees
# dec0 = 45.0     # Dec of the point of interest in degrees

# v_north, v_east = calculate_projections(cmb_ra, cmb_dec, ra0, dec0)
# print(f"Projection on North: {v_north:.6f}")
# print(f"Projection on East: {v_east:.6f}")
