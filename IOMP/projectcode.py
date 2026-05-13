# =====================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================
# 📂 Load dataset
# =====================================================
df = pd.read_csv(r"C:\Users\CHANDANA SAI\Desktop\IOMP\fatty liver dataset.csv")

print("\nMissing values:\n", df.isnull().sum())
df = df.dropna()

# =====================================================
# 🔤 Encoding
# =====================================================
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

print("\nDataset after encoding gender:")
print(df.head())

# =====================================================
# Create Age Groups
# =====================================================
df['age_group'] = pd.cut(df['age'], bins=[0,35,60,100], labels=['Young','Middle','Senior'])

print("\nAge Groups Created:")
print(df[['age','age_group']].head(10))

# Age Group Table
age_group_table = pd.DataFrame({
    "Age":["0–35","36–60","61–100"],
    "Group":["Young","Middle","Senior"]
})

print("\nAge Group Classification:")
print(age_group_table.to_markdown(index=False))

print("\nNumber of people in each age group:")
print(df['age_group'].value_counts())

age_risk = df.groupby('age_group')['is_patient'].sum()
print("\nNumber of patients in each age group:")
print(age_risk)

# Encode age_group
df['age_group'] = le.fit_transform(df['age_group'])

# =====================================================
# 📊 Correlation BEFORE scaling
# =====================================================
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Before Scaling")
plt.show()

# =====================================================
# ⚖️ Scaling
# =====================================================
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# AFTER scaling
plt.figure(figsize=(8,6))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
plt.title("After Scaling")
plt.show()

# =====================================================
# ✂️ Train-Test Split
# =====================================================
X = df_scaled.drop(columns=['is_patient'])
y = (df_scaled['is_patient'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# =====================================================

# =====================================================
# ⚖️ SCALING (ONLY FEATURES)
# =====================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# =====================================================
# ✂️ STRATIFIED SPLIT (BETTER GENERALIZATION)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# =====================================================
# ⚖️ SMOTE (FULL BALANCE)
# =====================================================
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# =====================================================
# 🔁 CROSS VALIDATION
# =====================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================================================
# 🚀 RANDOM FOREST (STRONG MODEL)
# =====================================================
from sklearn.ensemble import RandomForestClassifier

rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        'n_estimators':[300,500],
        'max_depth':[10,15,20,None],
        'min_samples_split':[2,5],
        'min_samples_leaf':[1,2]
    },
    cv=cv,
    n_jobs=-1
)

rf.fit(X_train_sm, y_train_sm)
rf_pred = rf.predict(X_test)

# =====================================================
# 🚀 XGBOOST 
# =====================================================
try:
    from xgboost import XGBClassifier

    xgb = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        {
            'n_estimators':[300,500],
            'max_depth':[3,5,7],
            'learning_rate':[0.03,0.05],
            'subsample':[0.8,1]
        },
        cv=cv,
        n_jobs=-1
    )

    xgb.fit(X_train_sm, y_train_sm)

    y_prob = xgb.predict_proba(X_test)[:,1]

    # 🔥 Threshold tuning (KEY TRICK)
    xgb_pred = (y_prob > 0.35).astype(int)

    xgb_available = True

except:
    print("⚠️ XGBoost not installed")
    xgb_available = False

# =====================================================
# 📊 RESULTS
# =====================================================
print("\n===== FINAL RESULTS =====")

print("\n🔹 Random Forest")
print("Accuracy:", round(accuracy_score(y_test, rf_pred)*100,2), "%")
print(classification_report(y_test, rf_pred))


# =====================================================
# 📉 CONFUSION MATRIX
# =====================================================
best_pred = xgb_pred if xgb_available else rf_pred

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# =====================================================
# 💾 SAVE MODEL
# =====================================================
best_model = xgb.best_estimator_ if xgb_available else rf.best_estimator_

pickle.dump(best_model, open("fatty_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl","wb"))

print("\n✅ Model saved successfully!")
