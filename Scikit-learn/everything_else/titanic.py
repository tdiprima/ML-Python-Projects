# Load titanic dataset
# Note: you don't get names with this
import seaborn as sns

titanic = sns.load_dataset('titanic')

print(titanic.head())
print(type(titanic))
