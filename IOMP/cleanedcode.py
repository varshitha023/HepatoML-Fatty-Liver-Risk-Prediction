import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"C:/Users/CHANDANA SAI/Desktop/IOMP/fatty liver dataset.csv")

print("Dataset Info:")

# -----------------------------
# Check missing values
# -----------------------------
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nShape BEFORE removing missing rows:", df.shape)

df = df.dropna()

print("Shape AFTER removing missing rows:", df.shape)

print(df.info())

# -----------------------------
# Encode Gender column
# -----------------------------
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

print("\nDataset after encoding gender:")
print(df.head())

# -----------------------------
# Create 3 Age Groups
# -----------------------------
df['age_group'] = pd.cut(df['age'],bins=[0, 35, 60, 100],labels=['Young', 'Middle', 'Senior'])

print("\nAge Groups Created:")
print(df[['age','age_group']].head(10))

# -----------------------------
# Age Group Classification Table
# -----------------------------
age_group_table = pd.DataFrame({"Age": ["0 – 35", "36 – 60", "61 – 100"],"Group": ["Young", "Middle", "Senior"]})

print("\nAge Group Classification:")
print(age_group_table.to_markdown(index=False))

# -----------------------------
# Count people in each group
# -----------------------------
print("\nNumber of people in each age group:")
print(df['age_group'].value_counts())

# -----------------------------
# Risk per age group
# -----------------------------
age_risk = df.groupby('age_group')['is_patient'].sum()

print("\nNumber of patients in each age group:")
print(age_risk)

# -----------------------------
# Encode age_group for correlation
# -----------------------------
df['age_group'] = le.fit_transform(df['age_group'])

# -----------------------------
# Correlation Heatmap BEFORE scaling
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap Before Scaling")
plt.show()

# -----------------------------
# Standard Scaling
# -----------------------------
scaler = StandardScaler()

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nScaled Dataset Sample:")
print(df_scaled.head())

# -----------------------------
# Correlation Heatmap AFTER scaling
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap After Scaling")
plt.show()

# =====================================================
# Train-Test Split
# =====================================================

from sklearn.model_selection import train_test_split

X = df_scaled.drop(columns=['is_patient'])
y = (df_scaled['is_patient'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================
# Import Metrics
# =====================================================

from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# Logistic Regression
# =====================================================

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("\nLogistic Regression Classification Report:\n")
print(classification_report(y_test, lr_pred))

# =====================================================
# Random Forest
# =====================================================

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_pred))

# =====================================================
# Support Vector Machine (SVM)
# =====================================================

from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=10)

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Classification Report:\n")
print(classification_report(y_test, svm_pred))

# =====================================================
# XGBoost
# =====================================================

from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,random_state=42)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("\nXGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("\nXGBoost Classification Report:\n")
print(classification_report(y_test, xgb_pred))