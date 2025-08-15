import glob
import numpy as np
import pandas as pd
from .LensmodelWrapper.utils import pandas_to_model
import os 

module_dir = os.path.dirname(os.path.abspath(__file__))
#list with the info for the different filters, this can be add to LensedQsoCensus

#filter_info =  pd.read_csv(f"{module_dir}/tables/filter_unique.csv")

class LensedQsoCensus:
    def __init__(self, path="Tables/photometry"):
        self.path_lens_tables = glob.glob(f"{module_dir}/{path}/*")
        #print(self.path_lens_tables)
        self.lens_census = self._do_full_lens_census()
        self.lens_census["year"] = [i[0:4] for i in self.lens_census["Bibcode"]]
        self.points = self.lens_census[["ra", "dec"]].values
        self.distances = np.linalg.norm(self.points[:, np.newaxis] - self.points, axis=2)

    def _do_full_lens_census(self):
        # Concatenate all tables into a single DataFrame
        return pd.concat(
            [pd.read_csv(file).assign(file=file) for file in self.path_lens_tables]
        ).reset_index(drop=True)

    def look_for_a_system(self, word, column="name", separation_limit=0.1):
        # Create a mask for non-NaN values and those containing the word
        #TODO add a filter for the columns i want to see 
        mask = self.lens_census[column].notna() & self.lens_census[column].astype("str").str.contains(str(word), regex=False)
        pandas_obj = self.lens_census[mask]
        
        if pandas_obj.empty:
            raise ValueError("Object not found in census.")
        
        # Calculate distance-based mask if needed
        if column in ["name", "z_s"]:
            idx = pandas_obj.index[0]
            close_mask = self.distances[idx] < separation_limit
            mask |= close_mask
        
        # Filter the data based on the combined mask
        filtered_data = self.lens_census[mask].dropna(axis=1, how='all')
        
        return filtered_data

    def unique_systems_count(self,get_unique_names=False,get_unique_years=False):
        # Initialize a mask for visited systems
        visited = np.zeros(len(self.lens_census), dtype=bool)
        count = 0
        names = []
        years = []
        # Vectorized approach to count unique systems
        for idx in range(len(self.lens_census)):
            if not visited[idx]:
                # Mark all systems within the separation limit as visited
                close_mask = self.distances[idx] < 0.1
                visited |= close_mask
                count += 1
                if get_unique_names:
                    names.append(self.lens_census.name[close_mask].drop_duplicates().values[0])
                if get_unique_years:
                    years.append([self.lens_census.name[close_mask].drop_duplicates().values[0],min(self.lens_census.year[close_mask].drop_duplicates().values.astype(int))])
        if get_unique_names:
            return names
        if get_unique_years:
            return years
        return count
    
    def hierarchical_selection(self,name):
        system = self.look_for_a_system(name).sort_values("year", ascending=False)
        pandas_s = np.array([[i,len(system[system["Bibcode"]==i])] for i in system.Bibcode.drop_duplicates().values])
        n_ = np.argmax(pandas_s[:,1].astype(int))
        data_to_model = system[system["Bibcode"]==system.Bibcode.drop_duplicates().values[n_]].dropna(axis=1, how='all').copy()
        if 'z_l' in system.columns and 'Bibcode' in system.columns:
            zl = [[zl, bibcode] for zl, bibcode in system[["z_l", "Bibcode"]].drop_duplicates().values if not pd.isnull(zl)][0]
            data_to_model[["z_l","Bibcode_zl"]] = [zl]* len(data_to_model)
        # Process the source redshift 'z_s'
        if 'z_s' in system.columns and 'Bibcode' in system.columns:
            zs = [[zs, bibcode] for zs, bibcode in system[["z_s", "Bibcode"]].drop_duplicates().values if not pd.isnull(zs)][0]
            data_to_model[["z_s","Bibcode_zs"]] = [zs] * len(data_to_model)
        data_to_model["known_names"] = [system.name.drop_duplicates().values]* len(data_to_model)
        data_to_model["can_be_modeled"] = [any([(("band" in col) and ("ima" in data_to_model["IS"].values)) for col in data_to_model.columns])] * len(data_to_model)
        return data_to_model
    # def pandas_for_model(self):
    #     pandas_to_be_use_in_model = pd.concat([pandas_to_model(self.hierarchical_selection(system_name)) for system_name in self.unique_systems_count(get_unique_names=True)]
    #     ).reset_index(drop=True)
    #     pandas_to_be_use_in_model['total_lens'] = pandas_to_be_use_in_model['total_lens'].fillna(0)
    #     return pandas_to_be_use_in_model

