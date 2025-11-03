import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
url = 'https://github.com/ad-1-t-1/FLOWSENSE/raw/main/Flood%20events%20India.xlsx'
df = pd.read_excel(url)

# 2. Explore Data
print("Columns:", df.columns)
print(df.info())
print(df.head())
print("Missing values:", df.isnull().sum())
print(df.describe())

# OPTIONAL: Rename columns for convenience if necessary
# df.rename(columns={'Flooded': 'flood'}, inplace=True)

# 3. Handle missing values (here we drop rows with missing)
df = df.dropna()

# 4. Define predictors and target
predictors = ['NDBI', 'NDVI', 'TWI', 'aspect', 'elevation', 'rainfall', 'slope']  # modify if needed!
target = 'flood'  # Make sure this matches your actual binary column name

X = df[predictors]
y = df[target]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling (for logistic regression; Random Forest/XGBoost don't require it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

# 8. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# Feature importance plot
plt.figure(figsize=(8,6))
sns.barplot(x=predictors, y=rf.feature_importances_)
plt.title('Random Forest Feature Importances')
plt.show()

# 9. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

# 10. ROC Curve comparison
from sklearn.metrics import roc_curve

plt.figure(figsize=(8,6))
for y_prob, label in [(y_prob_lr, 'Logistic'), (y_prob_rf, 'RF'), (y_prob_xgb, 'XGBoost')]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{label} (AUC: {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0,1],[0,1],'--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# 11. Calibration curve (for Random Forest as example)
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_prob_rf, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--', color='grey')
plt.title('Calibration plot (Random Forest)')
plt.xlabel('Mean predicted value')
plt.ylabel('Fraction of positives')
plt.show()

# --- End of notebook ---
