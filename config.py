'''Configuration module for project'''

import os


MAX_SAVED_CSV = 5


def available_loc(type_to_return="str"):
    '''
    Returns list of available locations,
    or string of most applicable loc for saving csv files in the data directory
    args:
        type_to_return: str, either "lst" or "str",
        determines the type of output returned by the function
    '''

    loc_lst = [f'{a}_smpl.csv' for a in range(1,MAX_SAVED_CSV+1)
               if f'{a}_smpl.csv' not in os.listdir(os.path.join(os.getcwd(), "data"))]

    return loc_lst if type_to_return == "lst" else loc_lst[0]




if __name__ == "__main__":
    pass
