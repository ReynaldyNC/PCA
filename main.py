from sklearn import datasets, tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load dataset
iris = datasets.load_iris()

# Separate attribute and label
x = iris.data
y = iris.target

# Divide dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define DecisionTree model
decision_tree = tree.DecisionTreeClassifier()

# Create PCA object with 2 principal component
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# Classifier accuracy test
model = decision_tree.fit(x_train_pca, y_train)

print(model.score(x_test_pca, y_test))
