import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# FRED APIキーを設定
api_key = "SECRET"
fred = Fred(api_key=api_key)

# 州のリストを作成
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

# 失業率と成長率のデータを取得するための空の辞書を作成
unemployment_data = {}
gdp_growth_data = {}

# 各州の失業率と成長率を取得
for state in states:
    # 失業率データを取得し、最新の値を辞書に格納
    unemployment_data[state] = fred.get_series(f"{state}UR")[-1]
    
    # GDPデータを取得し、前年比成長率を計算して辞書に格納
    gdp_data = fred.get_series(f"{state}RGSP")
    gdp_growth_data[state] = gdp_data[-1] / gdp_data[-2] - 1

# 取得したデータを使用してデータフレームを作成
data = pd.DataFrame({
    "State": states,
    "Unemployment Rate": list(unemployment_data.values()),
    "GDP Growth Rate": list(gdp_growth_data.values())
})

# KMeansアルゴリズムを使用してクラスタリングを実行
# n_clusters: クラスター数を指定 (ここでは4)
# random_state: 乱数シードを指定 (再現性のため)
kmeans = KMeans(n_clusters=4, random_state=0).fit(data[["Unemployment Rate", "GDP Growth Rate"]])

# クラスタリング結果をデータフレームに追加
data["Cluster"] = kmeans.labels_

# 結果を可視化するためのプロットを作成
plt.figure(figsize=(10, 6))

# 散布図を作成
# x軸: 失業率, y軸: GDP成長率, 色: クラスター, カラーマップ: viridis
plt.scatter(data["Unemployment Rate"], data["GDP Growth Rate"], c=data["Cluster"], cmap="viridis")

# x軸とy軸のラベルを設定
plt.xlabel("Unemployment Rate")
plt.ylabel("GDP Growth Rate")

# プロットのタイトルを設定
plt.title("Clustering of US States by Unemployment Rate and GDP Growth Rate")

# 各データポイントに州の略称を追加
for i, state in enumerate(data["State"]):
    plt.annotate(state, (data["Unemployment Rate"][i], data["GDP Growth Rate"][i]))

# カラーバーを追加
plt.colorbar(ticks=range(4), label="Cluster")

# プロットを表示
plt.show()
