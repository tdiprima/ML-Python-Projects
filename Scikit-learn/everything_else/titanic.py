# This downloads the dataset to ~/seaborn-data
# Note, you don't get names with this dataset
import seaborn as sns
import timeit

start_time = timeit.default_timer()
# Load titanic dataset
titanic = sns.load_dataset('titanic')  # Returns a pd.DataFrame
end_time = timeit.default_timer()

elapsed_time = end_time - start_time
print(f'Data loading took {elapsed_time} seconds to complete\n')

print(titanic.head())
