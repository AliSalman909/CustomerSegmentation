import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv(r"C:\Users\Ali's HP\Desktop\Mall_Customers.csv")
print(df.head())
print(df.info())

#renaming columns 
df.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']

sns.pairplot(df[['Age', 'AnnualIncome', 'SpendingScore']]) #draw pair wise scatterplots between age, income and score 
plt.show()

sns.heatmap(df[['Age', 'AnnualIncome', 'SpendingScore']].corr(), annot=True, cmap='coolwarm') #make correlate heatmap
plt.title("Feature Correlation")
plt.show()

X = df[['AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []  #checking inertia decreasing. less inertia is needed 
k_range = range(1, 11) #k from 1 to 10
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1')
plt.title("Customer Segments using PCA")
plt.show()

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE1'] = X_tsne[:, 0]
df['TSNE2'] = X_tsne[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=df, palette='Set2')
plt.title("Customer Segments using t-SNE")
plt.show()

for i in range(k):
    segment = df[df['Cluster'] == i]
    avg_income = segment['AnnualIncome'].mean()
    avg_score = segment['SpendingScore'].mean()
    print(f"Cluster {i}: Avg Income = {avg_income:.2f}, Avg Spending Score = {avg_score:.2f}")
