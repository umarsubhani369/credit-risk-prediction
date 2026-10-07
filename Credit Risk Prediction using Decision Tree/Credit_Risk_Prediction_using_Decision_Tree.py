import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix




df = pd.read_csv("credit_risk_dataset.csv")
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())
# df = df.drop_duplicated()

df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
# print(df.isnull().sum())

df = df.drop(columns=["person_age","loan_grade","loan_int_rate"])
# print(df.columns)

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# print(X.dtypes)
X = pd.get_dummies(X, columns=["person_home_ownership","loan_intent","cb_person_default_on_file"], drop_first=True)
# print(X.dtypes)
# print(X.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state = 42,stratify=y)


# for depth in [2,3,4,6,8,10, 12, 15, 20, None]:
#     model = DecisionTreeClassifier(
#         max_depth=depth,
#         random_state=42
#     )
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(
#         depth,
#         "Accuracy:", round(accuracy_score(y_test, y_pred), 3),
#         "Precision:", round(precision_score(y_test, y_pred), 3),
#         "Recall:", round(recall_score(y_test, y_pred), 3),
#         "F1:", round(f1_score(y_test, y_pred), 3)
#     )



model = DecisionTreeClassifier(max_depth=15,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
# print(f"Accuracy:{accuracy}")
# print(f"Precision:{precision}")
# print(f"Recal:{recall}")
# print(f"f1_score:{f1}")
# print(cm)


person_income = float(input("Enter person income: "))
person_emp_length = float(input("Enter employment length: "))
loan_amnt = float(input("Enter loan amount: "))
loan_percent_income = float(input("Enter loan percent income: "))
cb_person_cred_hist_length = float(input("Enter credit history length: "))

person_home_ownership = input("Enter home ownership (RENT/OWN/OTHER): ").upper()
loan_intent = input("Enter loan intent (EDUCATION/HOMEIMPROVEMENT/MEDICAL/PERSONAL/VENTURE): ").upper()
cb_person_default_on_file = input("Default on file? (Y/N): ").upper()


user_input = pd.DataFrame([{
    "person_income": person_income,
    "person_emp_length": person_emp_length,
    "loan_amnt": loan_amnt,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "person_home_ownership": person_home_ownership,
    "loan_intent": loan_intent,
    "cb_person_default_on_file": cb_person_default_on_file
}])


user_input = pd.get_dummies(
    user_input,
    columns=[
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file"
    ],
    drop_first=True
)


user_input = user_input.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(user_input)

if prediction[0] == 0:
    print("Low Risk - No Default")
else:
    print("High Risk - Default")