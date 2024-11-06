#pip install -r requirements.txt
import sys
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))
inertias = []

for i in range(1, 11):
    # Indent the code block within the for loop
    kmean = KMeans(n_clusters=i)
    kmean.fit(data)
    inertias.append(kmean.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()