import numpy as np

def compute_directional_gini(model,normalize=True):
    values = model.tree_.value.T[0][0]
    left_c = model.tree_.children_left
    right_c = model.tree_.children_right

    impurity = model.tree_.impurity
    node_samples = model.tree_.weighted_n_node_samples

    feature_importance = np.zeros((model.tree_.n_features,))
    directional_feature_importance = np.zeros((model.tree_.n_features,))
    for idx,node in enumerate(model.tree_.feature):
        if node >= 0:
            left_value = values[left_c[idx]]
            right_value = values[right_c[idx]]
            diff = right_value - left_value
            diff /= np.abs(diff)
            feature_importance[node]+= (impurity[idx]*node_samples[idx]- \
                                   impurity[left_c[idx]]*node_samples[left_c[idx]]-\
                                   impurity[right_c[idx]]*node_samples[right_c[idx]])
            directional_feature_importance[node]+= diff *(impurity[idx]*node_samples[idx]- \
                                   impurity[left_c[idx]]*node_samples[left_c[idx]]-\
                                   impurity[right_c[idx]]*node_samples[right_c[idx]])

    directional_feature_importance/=node_samples[0]
    feature_importance /= node_samples[0]
    if normalize:
        directional_feature_importance /= np.sum(np.abs(feature_importance))

    return directional_feature_importance

def compute_ensemble_directional_gini(model):
    outputs = []
    for tree in model.estimators_:
        output = compute_directional_gini(tree)
        outputs.append(output)
    return np.mean(np.array(outputs),axis=0)
