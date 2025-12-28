#טעינת ושמירת הדאטה
import pandas as pd
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()

#בדיקת שמות המשתנים  
print(df.columns.to_list())

import seaborn as sns
import matplotlib.pyplot as plt

#יצירת הגרפים לשם בחירת המשתנים האידאליים לחיזוי 
sns.pairplot(df, hue='status', diag_kind= "kde", corner= True)
plt.show()

#נרמול המשתנים הנבחרים לטווח של 1-0
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(df[['PPE', 'DFA']])

#פיצול הדאטה לאימון ובדיקה
from sklearn.model_selection import train_test_split

X = x
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#בחירת מודל אימון זהה לכותבי המאמר
from sklearn.svm import SVC

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

#בדיקת דיוק המודל
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#הורדת המודל
import joblib

joblib.dump(model, 'my_model.joblib')

