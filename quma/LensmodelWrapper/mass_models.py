def mass_models(mass_distribution,n_galaxies, p1, lens_centered, path):
    if mass_distribution == "SIS":
            # Initialize l as an empty string
            model = f"alpha {p1} 0 0 0.0 0.0 0 0.0 0 0 1\n"
            if n_galaxies > 1:
                for n in range(1, n_galaxies):
                    ra, dec = lens_centered[n]
                    # Create the line with p1/2 and the current ra, dec
                    model += f"alpha {p1 / 2} {ra} {dec} 0.0 0.0 0 0.0 0 0 1\n"
                    #l += line  # Concatenate the line to l
            # Add the fixed line
            model += "1 1 1 0 0 0 0 0 0 0\n"
            if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
            # Add the optimize line with the given path
            model += f"optimize {path}/final_step\n"
    elif  mass_distribution in ["SIE","POW"]:
            # Initialize l as an empty string
            model = f"alpha {p1} 0 0 0.003 10.0 0 0.0 0 0 1\n"
            if n_galaxies > 1:
                for n in range(1, n_galaxies):
                    ra, dec = lens_centered[n]
                    # Create the line with p1/2 and the current ra, dec
                    model += f"alpha {p1 / 2} {ra} {dec} 0.0 0.0 0 0.0 0 0 1\n"
                    #l += line  # Concatenate the line to l
            # Add the fixed line
            model += "1 0 0 1 1 0 0 0 0 0\n"
            if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
            model += f"varyone 1 5 -90.0 90.0 19 {path}/step_1\n"
            model += f"\nsetlens {path}/step_1.start\nchangevary 1"#heer maybe is the number of galaxies  
            model += f"\n1 0 0 1 1 0 0 0 0 0 \n"
            if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 1 1 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
            model += f"optimize {path}/step_2\n"
            if mass_distribution=="SIE":
                model += f"\nset chimode= 1\nsetlens {path}/step_2.start\nchangevary 1\n1 1 1 1 1 0 0 0 0 0\n"
            elif mass_distribution=="POW":
                model += f"\nset chimode= 1\nsetlens {path}/step_2.start\nchangevary 1\n1 1 1 1 1 0 0 0 0 1\n"
            if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
            model += f"\noptimize {path}/final_step\n"
    elif "SIS+shear" == mass_distribution:
        model = f"alpha {p1} 0 0 0.0 0.0 0.003 10.0 0 0 1\n"
        if n_galaxies > 1:
                for n in range(1, n_galaxies):
                    ra, dec = lens_centered[n]
                    # Create the line with p1/2 and the current ra, dec
                    model += f"alpha {p1 / 2} {ra} {dec} 0.0 0.0 0 0.0 0 0 1\n"
                    #l += line  # Concatenate the line to l
        model+= "1 0 0 0 0 1 1 0 0 0\n"
        if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
        model+= f"varyone 1 7 -90.0 90.0 19 {path}/step_1\n"
        model+= f"setlens {path}/step_1.start\n"
        model+= f"changevary 1 \n1 0 0 0 0 1 1 0 0 0\n"
        if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
        model+= f"\noptimize {path}/step_2\n"
        model += f"set chimode= 1\nsetlens {path}/step_2.start\n"
        model += f"changevary 1\n"
        model += "1 1 1 0 0 1 1 0 0 0\n"
        if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
        model += f"optimize {path}/final_step\n"
    elif mass_distribution in ["SIE+shear","POW+shear"]:
        model = f"alpha {p1} 0 0 0.03 10.0 0 0.0 0 0 1\n"
        if n_galaxies > 1:
                for n in range(1, n_galaxies):
                    ra, dec = lens_centered[n]
                    # Create the line with p1/2 and the current ra, dec
                    model += f"alpha {p1} {ra} {dec} 0.0 0.0 0 0.0 0 0 1\n"
                    #l += line  # Concatenate the line to l
        model+= "1 0 0 1 1 0 0 0 0 0\n"
        model+= f"varytwo 1 5 -90.0 90.0 19 1 7 -90.0 90.0 19 {path}/step_1\n"
        model+= f"setlens {path}/step_1.start\n"
        model+= f"changevary 1 \n1 1 1 1 1 0 0 0 0 0\n"
        if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
        model+= f"\noptimize {path}/step_2\n"
        model += f"set chimode= 1\nsetlens {path}/step_2.start\n"
        model += f"changevary 1\n"
        if "SIE+shear" ==mass_distribution:
            model += "1 1 1 1 1 1 1 0 0 0\n"
        elif "POW+shear" ==mass_distribution:
            model += "1 1 1 1 1 1 1 0 0 1\n"
        if n_galaxies > 1:
                for _ in range(1, n_galaxies):
                    model += "1 0 0 0 0 0 0 0 0 0\n"  # Concatenate the fixed line multiple times
        model += f"optimize {path}/final_step\n"
    else: 
        raise print(mass_distribution,"not avalaible")
    model += f"set plotmode    = 2\nkapgam 3 {path}/kappa_gamma.dat\nplotcrit {path}/critical_caustic.dat\nplotgrid {path}/grid.dat\ncalcRein 10 {path}/RE.dat\nquit"    
    return model