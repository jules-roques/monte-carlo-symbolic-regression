# 📊 Ajouts graphiques (ultra concis)

## 0. Pré-requis

Ajouter en haut du fichier :

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 1. Ajouter le nom de méthode dans les résultats

Dans `run_single_problem`, ajouter :

```python
"method": config.get("algorithm", {}).get("class_name", "Unknown"),
```

---

## 2. Construire un DataFrame global

Après `results` :

```python
df = pd.DataFrame(results)
```

---

# 🔥 3. Heatmap (équation × méthode)

```python
pivot = df.pivot(index="name", columns="method", values="val_r2")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap="viridis")
plt.title("Heatmap Val R2")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()
```

---

# 📊 4. Grouped bar chart (par équation)

```python
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="name", y="val_r2", hue="method")
plt.xticks(rotation=45)
plt.title("Val R2 par équation")
plt.tight_layout()
plt.savefig("barplot_grouped.png")
plt.close()
```

---

# ⚡ 5. Scatter plot (val_r2 vs temps)

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="elapsed_seconds", y="val_r2", hue="method")
plt.title("Performance vs Temps")
plt.tight_layout()
plt.savefig("scatter_time_vs_r2.png")
plt.close()
```

---

# ✅ 6. Où placer ça

Ajouter juste après :

```python
print_results_table(results)
```

---

# ⚠️ Important

Si tu compares plusieurs méthodes :

* concatène plusieurs fichiers JSON avant (`pd.concat`)
* sinon la heatmap aura une seule colonne
