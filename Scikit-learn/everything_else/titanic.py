"""
Loads the Titanic dataset from Seaborn, measures how long it takes to load,
and prints the time taken and the first five rows of the dataset.
"""
import seaborn as sns
import timeit

start_time = timeit.default_timer()
# Load titanic dataset
titanic = sns.load_dataset('titanic')  # Returns a pd.DataFrame
end_time = timeit.default_timer()

elapsed_time = end_time - start_time
print(f'Data loading took {elapsed_time} seconds to complete\n')

print(titanic.head())
