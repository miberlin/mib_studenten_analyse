from utils import *
# read the configuration file and initialize random generators
cfg = read_config('config.yaml')

# define variables with config file

# generate dataframes
df_termine, df_studenten, df_studentenxtermine = generate_dataframes(cfg)
# print(df_studentenxtermine.info())