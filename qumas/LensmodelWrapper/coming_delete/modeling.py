def models_disp(imagenes:list,dif:list):
    #all use SIE
    Models=[]
    if len(imagenes)==1:
        Models=["SIS"]
    if len(imagenes)==2:
        Models=["SIS","SIE","SIS+shear"]
    if len(imagenes)==3:
        Models=["SIS","SIE","SIS+shear"]
    if len(imagenes)==4:
        Models=["SIE","SIE+shear"]
        if len(dif)>1:
            Models.append("SIE-2G")
    if len(imagenes)>=5 and len(dif)>1:
        Models.append("SIE-2G")
    return Models
def writte_lensmodel_file():
    return