from numpy import square

from knn import KNN
from myutils import split_data, load_csv_data


IRIS = load_csv_data("knn/data/iris.csv")
DIABETES = load_csv_data("knn/data/diabetes.csv")

iris_data, iris_target = split_data(IRIS).values()
diabetes_data, diabetes_target = split_data(DIABETES).values()

print()
print()

print("PRINTING THE ACCURACY OF IRIS KNN CLASSIFIER")
print("----------------------------------------------------------------------")

k_values = [1, 3, 5, 10, 15]

for k in k_values:
    iris_knn = KNN(iris_data["train"], iris_target["train"], k_neighbors=k)

    iris_val_count = len(iris_target["validation"])
    iris_correct_count = 0

    for i in range(iris_val_count):
        if (
            iris_knn.predict_class(iris_data["validation"][i])
            == iris_target["validation"][i]
        ):

            iris_correct_count += 1

    accuracy = (iris_correct_count / iris_val_count) * 100
    print(f"for k = {k:<2d}, accuracy = {accuracy:<.2f}%")

print()

print("Test Data:")

iris_knn = KNN(iris_data["train"], iris_target["train"], k_neighbors=5)

iris_val_count = len(iris_target["test"])
iris_correct_count = 0

for i in range(iris_val_count):
    if iris_knn.predict_class(iris_data["test"][i]) == iris_target["test"][i]:
        iris_correct_count += 1

accuracy = (iris_correct_count / iris_val_count) * 100

print(f"For k = 5, Accuracy = {accuracy:.2f}%")
print()
print()

print("PRINTING THE MEAN SQUARE ERROR OF DIABETES KNN REGRESSOR")
print("----------------------------------------------------------------------")

for k in k_values:
    diabetes_knn = KNN(diabetes_data["train"], diabetes_target["train"], k_neighbors=k)

    diabetes_val_count = len(diabetes_target["validation"])
    square_sum = 0

    for i in range(diabetes_val_count):
        square_diff = square(
            diabetes_knn.predict_regressed(diabetes_data["validation"][i])
            - diabetes_target["validation"][i]
        )
        square_sum += square_diff
    mse = square_sum / diabetes_val_count
    print(f"for k = {k:<2d}, MSE = {mse:<.2f}")

print()

print("Test Data:")

diabetes_knn = KNN(diabetes_data["train"], diabetes_target["train"], k_neighbors=10)

diabetes_val_count = len(diabetes_target["test"])
square_sum = 0

for i in range(diabetes_val_count):
    square_diff = square(
        diabetes_knn.predict_regressed(diabetes_data["test"][i])
        - diabetes_target["test"][i]
    )
    square_sum += square_diff
mse = square_sum / diabetes_val_count

print(f"For k = 10, MSE = {mse:.2f}")

print()
print()
