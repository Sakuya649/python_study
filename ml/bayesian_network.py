import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# データの作成（カテゴリカルな変数）
data = {'Age': ['Young', 'Middle', 'Old', 'Middle', 'Young'],
        'Income': ['Low', 'High', 'Medium', 'High', 'Low'],
        'Buy': ['No', 'Yes', 'Yes', 'Yes', 'No']}
df = pd.DataFrame(data)

# モデルの定義
model = BayesianNetwork([('Age', 'Buy'), ('Income', 'Buy')])

# 条件付き確率分布の定義
cpd_age = TabularCPD('Age', 3, [[0.3], [0.4], [0.3]], state_names={'Age': ['Young', 'Middle', 'Old']})  # 年齢の事前分布
cpd_income = TabularCPD('Income', 3, [[0.3], [0.4], [0.3]], state_names={'Income': ['Low', 'Medium', 'High']})  # 収入の事前分布
cpd_buy = TabularCPD('Buy', 2, [[0.2, 0.5, 0.8, 0.3, 0.4, 0.6, 0.1, 0.2, 0.4],  # Buy = "No"
                                [0.8, 0.5, 0.2, 0.7, 0.6, 0.4, 0.9, 0.8, 0.6]],  # Buy = "Yes"
                    evidence=['Age', 'Income'],
                    evidence_card=[3, 3],
                    state_names={'Buy': ['No', 'Yes'], 'Age': ['Young', 'Middle', 'Old'], 'Income': ['Low', 'Medium', 'High']})

# モデルへの追加
model.add_cpds(cpd_age, cpd_income, cpd_buy)

# 推論
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
query = infer.query(variables=['Buy'], evidence={'Age': 'Young', 'Income': 'High'})
print(query)

import networkx as nx
import matplotlib.pyplot as plt

# グラフの作成
G = nx.DiGraph()
G.add_nodes_from(['Buy', 'Age', 'Income'])
G.add_edges_from([('Age', 'Buy'), ('Income', 'Buy')])

# レイアウトの決定
pos = nx.spring_layout(G)

# 図の描画
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()