import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)
    
    # сортируем по значению признака
    sorted_indices = np.argsort(feature_vector)
    x_sorted = feature_vector[sorted_indices]
    y_sorted = target_vector[sorted_indices]
    
    unique_values = np.unique(x_sorted)
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    thresholds_matrix = thresholds[:, np.newaxis]
    x_matrix = x_sorted[np.newaxis, :]

    left_mask = x_matrix < thresholds_matrix
    right_mask = ~left_mask

    n_total = len(target_vector)

    left_counts = np.sum(left_mask, axis=1)
    right_counts = n_total - left_counts

    valid = (left_counts > 0) & (right_counts > 0)
    if not np.any(valid):
        return None, None, None, None  # Нет подходящих разбиений
    
    y_matrix = y_sorted[np.newaxis, :]
    left_positive = np.sum(y_matrix * left_mask, axis=1)
    right_positive = np.sum(y_matrix * right_mask, axis=1)

    left_total = left_counts
    right_total = right_counts

    left_p1 = np.divide(left_positive, left_total, out=np.zeros_like(left_positive, dtype=float), where=left_total != 0)
    right_p1 = np.divide(right_positive, right_total, out=np.zeros_like(right_positive, dtype=float), where=right_total != 0)


    left_p0 = 1 - left_p1
    right_p0 = 1 - right_p1

    H_left = 1 - left_p1 ** 2 - left_p0 ** 2
    H_right = 1 - right_p1 ** 2 - right_p0 ** 2

    Q = - (left_total / n_total) * H_left - (right_total / n_total) * H_right

    valid = (left_total > 0) & (right_total > 0)
    thresholds_valid = thresholds[valid]
    ginis_valid = Q[valid]

    # Выбираем лучший по минимальному значению Джини (и минимальному порогу в случае равенства)
    best_idx = np.argmin(ginis_valid) 
    threshold_best = thresholds_valid[best_idx]
    gini_best = ginis_valid[best_idx]

    return thresholds_valid, ginis_valid, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, cur_depth):
        n_objects = len(sub_y)
    
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
    
        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}
    
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = 0 if current_click == 0 else current_count / current_click
                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
    
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
    
            if threshold is None:
                continue
    
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
    
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [x[0] for x in filter(lambda x: x[1] < threshold, categories_map.items())]
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")
    
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
    
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError(f"Unknown feature type: {self._feature_types[feature_best]}")
    
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], cur_depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], cur_depth + 1)

    
    def get_depth(self):
        return self._get_depth_recursive(self._tree)
    
    def _get_depth_recursive(self, node):
        if node["type"] == "terminal":
            return 0
        
        left_depth = self._get_depth_recursive(node["left_child"])
        right_depth = self._get_depth_recursive(node["right_child"])
        
        return 1 + max(left_depth, right_depth)
    
    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
    
        feature_index = node["feature_split"]
        feature_value = x[feature_index]
    
        if self._feature_types[feature_index] == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_index] == "categorical":
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")
        
    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree, cur_depth=0) 


    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
