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
    v_cmb = np.tile(ra_dec_to_unit_vector(cmb_ra, cmb_dec)[:,np.newaxis], ra0.shape).T
    #print(v_cmb.shape)
    # Calculate the unit vector for North direction
    dec0_rad = np.radians(dec0)
    ra0_rad = np.radians(ra0)
    
    v_north = np.array([-np.sin(dec0_rad) * np.cos(ra0_rad),
                        -np.sin(dec0_rad) * np.sin(ra0_rad),
                        np.cos(dec0_rad)])
    #print(v_north.shape)
    # Calculate the unit vector for East direction
    v_east = np.array([-np.sin(ra0_rad),
                        np.cos(ra0_rad),
                        np.zeros(ra0_rad.shape)])
    #print(v_east.shape)
    # Project the CMB dipole onto the North and East directions
    projection_north = np.dot(v_cmb, v_north)
    #print(projection_north.shape)
    projection_east = np.dot(v_cmb, v_east)
    #print(projection_east.shape)
    return np.diag(projection_north), np.diag(projection_east)