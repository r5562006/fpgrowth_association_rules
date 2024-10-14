# fpgrowth_association_rules.py
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 生成隨機數據
data = pd.DataFrame(np.random.randint(0, 2, size=(100, 5)), columns=['A', 'B', 'C', 'D', 'E'])

# 應用 FP-Growth 算法
frequent_itemsets = fpgrowth(data, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules)