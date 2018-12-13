from math import log2
from math import sqrt
from random import randint
from queue import PriorityQueue

import numpy as np
from sklearn.metrics import confusion_matrix


class TreeNode():
    def __init__(self, attr_list_for_split, depth, is_leaf=False, attr=None, continuous_attr_value=None, parent=None,
                 classification=None, is_root=False):
        self.attr_list_for_split = attr_list_for_split  # the currently remained attributes that can used for building tree
        self.depth = depth
        self.is_leaf = is_leaf
        self.attr = attr  # the chosen attribute for growing sub-trees
        self.continuous_attr_value = continuous_attr_value  # if self.attr is continuous, record the split value of it
        self.parent = parent
        self.classification = classification  # classification result for leaf node
        self.descendants = {}  # dict, keys for attr values, values for child nodes
        self.y_frequence_map = None  # counts the occurrence number of all kinds of y, used in method 'pruning'
        self.is_root = is_root

    # used for TreeNode sort in pruning process
    def __lt__(self, other):
        return self.depth > other.depth


class DTC45():
    def __init__(self, max_depth=35, min_samples_split=2, max_continuous_attr_splits=150, building_random_forest=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_continuous_attr_splits = max_continuous_attr_splits
        self.building_random_forest = building_random_forest
        print("Tree parameter settings: max_depth={}, min_samples_split={}, max_continuous_attr_splits={}".format(
            max_depth, min_samples_split, max_continuous_attr_splits))
        self.root = None
        self.built = False  # only after the tree is built, can pruning be performed

    def fit(self, X_train, y_train, attr_list, attr_is_discrete, attr_discrete_values=None, verbose=0):

        self.y_kinds = set(y_train)
        self.attr_list = attr_list
        self.attr_position_map = dict(zip(attr_list, range(len(attr_list))))
        # if the elements in *attr_is_discrete* are [True, False] format, convert them to [1, 0] format
        if attr_is_discrete[0] in [True, False]:
            attr_is_discrete = list([1 if x==True else 0 for x in attr_is_discrete])
        self.attr_is_discrete_map = dict(zip(attr_list, attr_is_discrete))

        if attr_discrete_values != None:
            self.attr_discrete_values = attr_discrete_values
        else:
            self.attr_discrete_values = {}
            for i in range(len(attr_list)):
                attr = attr_list[i]
                if self.attr_is_discrete_map[attr]:
                    self.attr_discrete_values[attr] = set([x[i] for x in X_train])

        self.root = TreeNode(attr_list_for_split=attr_list, depth=0, is_root=True)
        if verbose:
            print('# Building tree, wait please...')
            print('Number of samples used for building:', len(X_train))
        self.root = self._build_tree(self.root, X_train, y_train, verbose=verbose)
        self.built = True

    def _build_tree(self, node, X_train, y_train, verbose=0):

        if verbose:
            print("Sub-tree size: {}, depth: {},".format(len(y_train), node.depth))

        assert len(y_train) > 0

        # all left samples have the same classification result
        cur_y_all_same = True
        cur_y = y_train[0]
        for y in y_train[1:]:
            if y != cur_y:
                cur_y_all_same = False
                break
        if cur_y_all_same:
            if verbose:
                print("Set leaf node for y values are all same.")
            node.is_leaf = True
            node.classification = cur_y
            return node

        # all leaf samples have the same values on the leaf attributes for building tree
        cur_X_all_same = True
        for attr in node.attr_list_for_split:
            if cur_X_all_same:
                attr_pos = self.attr_position_map[attr]
                attr_v = X_train[0][attr_pos]
                for x in X_train[1:]:
                    if x[attr_pos] != attr_v:
                        cur_X_all_same = False
                        break
            else:
                break
        if cur_X_all_same:
            if verbose:
                print("Set leaf node for X values are all same on the current leaf attributes.")
            node.is_leaf = True
            node.classification = self._find_most_frequent_value(y_train)
            return node

        # the attribute list for split is empty
        if (len(node.attr_list_for_split) == 0):
            if verbose:
                print("Set leaf node for empty attri_list to split.")
            node.is_leaf = True
            node.classification = self._find_most_frequent_value(y_train)
            return node

        # the maximum of tree depth is reached
        if (node.depth == self.max_depth):
            if verbose:
                print("Set leaf node for maximal tree depth is reached.")
            node.is_leaf = True
            node.classification = self._find_most_frequent_value(y_train)
            return node

        # the number of left samples is less than the minimum split threshold
        if (len(X_train) < self.min_samples_split):
            if verbose:
                print("Set leaf node for minimal sample split is reached.")
            node.is_leaf = True
            node.classification = self._find_most_frequent_value(y_train)
            return node

        # find current best attribute for growth
        # if the best_attr is a continuous numeric attribute,
        # then best_attr_split_value will be the best binary split position of this attribute,
        # otherwise, the best_attr_split_value will be None
        best_attr, best_attr_split_value = self._choose_best_split_attr(X_train, y_train,
                                                                        attr_list=node.attr_list_for_split)
        if verbose:
            print("Best attr: {}, attr_value: {}".format(best_attr, best_attr_split_value))

        node.attr = best_attr
        if (not self.attr_is_discrete_map[best_attr]):
            node.continuous_attr_value = best_attr_split_value

        # build sub-trees
        # process discrete attribute
        if (self.attr_is_discrete_map[best_attr]):
            best_attr_values = self.attr_discrete_values[best_attr]
            descendants_X = {}
            descendants_y = {}
            for attr_value in best_attr_values:
                descendants_X[attr_value] = []
                descendants_y[attr_value] = []
            attr_pos = self.attr_position_map[best_attr]
            for i in range(len(X_train)):
                descendants_X[X_train[i][attr_pos]].append(X_train[i])
                descendants_y[X_train[i][attr_pos]].append(y_train[i])
            reduced_attr_list_for_split = node.attr_list_for_split[:]
            reduced_attr_list_for_split.remove(best_attr)
            for attr_v in best_attr_values:
                if len(descendants_X[attr_v]) > 0:
                    child_node = TreeNode(attr_list_for_split=reduced_attr_list_for_split,
                                          depth=node.depth + 1, parent=node)
                    node.descendants[attr_v] = self._build_tree(child_node, descendants_X[attr_v],
                                                                descendants_y[attr_v], verbose=verbose)
                else:
                    # no samples has this attribute value, so set child node to leaf node.
                    # for the classification result, use the most frequent y value in y_train
                    child_node = TreeNode(attr_list_for_split=reduced_attr_list_for_split,
                                          depth=node.depth + 1, parent=node,
                                          is_leaf=True, classification=self._find_most_frequent_value(y_train))
                    node.descendants[attr_v] = child_node

        # process continuous numeric attribute
        elif (not self.attr_is_discrete_map[best_attr]):
            best_attr_values = ['less', 'greater']
            descendants_X = {}
            descendants_y = {}
            for attr_value in best_attr_values:
                descendants_X[attr_value] = []
                descendants_y[attr_value] = []
            attr_pos = self.attr_position_map[best_attr]
            for i in range(len(X_train)):
                if X_train[i][attr_pos] < best_attr_split_value:
                    descendants_X['less'].append(X_train[i])
                    descendants_y['less'].append(y_train[i])
                else:
                    descendants_X['greater'].append(X_train[i])
                    descendants_y['greater'].append(y_train[i])
            for attr_v in best_attr_values:
                if len(descendants_X[attr_v]) > 0:
                    child_node = TreeNode(attr_list_for_split=node.attr_list_for_split, depth=node.depth + 1,
                                          parent=node)
                    node.descendants[attr_v] = self._build_tree(child_node, descendants_X[attr_v],
                                                                descendants_y[attr_v], verbose=verbose)
                else:
                    # no samples has this attribute value, so set child node to leaf node.
                    # for the classification result, use the most frequent y value in y_train
                    child_node = TreeNode(attr_list_for_split=node.attr_list_for_split, depth=node.depth + 1,
                                          parent=node,
                                          is_leaf=True, classification=self._find_most_frequent_value(y_train))
                    node.descendants[attr_v] = child_node

        return node

    def _find_most_frequent_value(self, values):

        value_number_map = {}
        for v in values:
            if v not in value_number_map:
                value_number_map[v] = 1
            else:
                value_number_map[v] += 1

        max_number = 0
        most_frequent_value = None
        for key in value_number_map:
            if value_number_map[key] > max_number:
                max_number = value_number_map[v]
                most_frequent_value = v

        if most_frequent_value is None:
            print("We've found a None y list !!")
            assert most_frequent_value != None

        return most_frequent_value

    def _choose_best_split_attr(self, X, y, attr_list, verbose=0):
        best_attr = None
        best_attr_gain_ratio = 0
        best_attr_split_value = None

        # calculate entropy value of current tree node
        y_occurrence = dict(zip(self.y_kinds, [0] * len(self.y_kinds)))
        for yi in y:
            y_occurrence[yi] = y_occurrence[yi] + 1
        ent = self._calculate_entropy_from_frequency(y_occurrence, len(y))

        # deal with each attribute
        X_y = [list(xi) + [yi] for (xi, yi) in zip(X, y)]

        # forming candidate attribute set
        if self.building_random_forest:
            # select int(log_{2}^{k}) random attributes
            attr_num = int(np.log2(len(attr_list)))
            if attr_num<1:
                attr_num = 1
            candidate_attrs = []
            while len(candidate_attrs)<attr_num:
                attr = attr_list(randint(0, len(attr_list)-1))
                if attr not in candidate_attrs:
                    candidate_attrs.append(attr)
        else:
            candidate_attrs = attr_list

        for attr in candidate_attrs:
            attr_pos = self.attr_position_map[attr]
            # two dimensional map for query the occurrences of given attribute value and y value
            attr_value_y = {}

            # for discrete attributes
            if (self.attr_is_discrete_map[attr]):
                attr_v_occurrence = dict(
                    zip(self.attr_discrete_values[attr], [0] * len(self.attr_discrete_values[attr])))
                for attr_v in self.attr_discrete_values[attr]:
                    attr_value_y[attr_v] = dict(zip(self.y_kinds, [0] * len(self.y_kinds)))
                for x_y in X_y:
                    # print(x_y)
                    attr_value_y[x_y[attr_pos]][x_y[-1]] = attr_value_y[x_y[attr_pos]][x_y[-1]] + 1
                    attr_v_occurrence[x_y[attr_pos]] = attr_v_occurrence[x_y[attr_pos]] + 1
                ent_attr = 0
                for attr_v in self.attr_discrete_values[attr]:
                    ent_attr = ent_attr + attr_v_occurrence[attr_v] / (len(y)) * self._calculate_entropy_from_frequency(
                        attr_value_y[attr_v], attr_v_occurrence[attr_v])
                intrinsic_value = self._calculate_entropy_from_frequency(attr_v_occurrence, len(y))

                # update best gain_ratio
                if intrinsic_value > 0:
                    gain_ratio = (ent - ent_attr) / intrinsic_value
                    if (gain_ratio > best_attr_gain_ratio):
                        best_attr = attr
                        best_attr_gain_ratio = gain_ratio


            # for continuous numeric attributes
            else:
                attr_split_value_list = list(set([x_y[attr_pos] for x_y in X_y]))
                attr_split_value_list.sort()
                # cut down the size of split value candidates
                while (len(attr_split_value_list) > self.max_continuous_attr_splits):
                    attr_split_value_list = [attr_split_value_list[i] for i in range(len(attr_split_value_list)) if
                                             i % 3 == 0]
                # process each split value of the current attribute
                for attr_split_value in attr_split_value_list:
                    attr_v_occurrence = dict(zip(['less', 'greater'], [0] * 2))
                    for attr_v in ['less', 'greater']:
                        attr_value_y[attr_v] = dict(zip(self.y_kinds, [0] * len(self.y_kinds)))
                    for x_y in X_y:
                        # print(x_y)
                        if x_y[attr_pos] < attr_split_value:
                            attr_value_y['less'][x_y[-1]] = attr_value_y['less'][x_y[-1]] + 1
                            attr_v_occurrence['less'] = attr_v_occurrence['less'] + 1
                        else:
                            attr_value_y['greater'][x_y[-1]] = attr_value_y['greater'][x_y[-1]] + 1
                            attr_v_occurrence['greater'] = attr_v_occurrence['greater'] + 1
                    ent_attr = 0
                    for attr_v in ['less', 'greater']:
                        ent_attr = ent_attr + attr_v_occurrence[attr_v] / (
                            len(y)) * self._calculate_entropy_from_frequency(
                            attr_value_y[attr_v], attr_v_occurrence[attr_v])
                    intrinsic_value = self._calculate_entropy_from_frequency(attr_v_occurrence, len(y))

                    # update best gain_ratio
                    if intrinsic_value > 0:
                        gain_ratio = (ent - ent_attr) / intrinsic_value
                        if (gain_ratio > best_attr_gain_ratio):
                            best_attr = attr
                            best_attr_gain_ratio = gain_ratio
                            if (not self.attr_is_discrete_map[attr]):
                                best_attr_split_value = attr_split_value

        if best_attr == None:
            if verbose:
                print("None best_attr found.")
            best_attr = attr_list[0]
            best_attr_split_value = X[0][self.attr_position_map[best_attr]]

        if self.attr_is_discrete_map[best_attr]:
            return best_attr, None
        else:
            return best_attr, best_attr_split_value

    def _calculate_entropy_from_frequency(self, y_occurrence_map, total_len):
        entropy = 0.0
        if total_len > 0:
            for yi in y_occurrence_map:
                prob = y_occurrence_map[yi] / total_len
                if prob > 0:
                    entropy = entropy - prob * log2(prob)
        return entropy

    def predict(self, X_test):
        if not self.built:
            print("You should build the tree first by calling the 'fit' method with some train samples.")
            return None

        y_predict = []
        for x in X_test:
            cur_node = self.root
            while (not cur_node.is_leaf):

                x_attr_value = x[self.attr_position_map[cur_node.attr]]
                if (self.attr_is_discrete_map[cur_node.attr]):
                    cur_node = cur_node.descendants[x_attr_value]
                else:
                    if x_attr_value < cur_node.continuous_attr_value:
                        cur_node = cur_node.descendants['less']
                    else:
                        cur_node = cur_node.descendants['greater']
            y_predict.append(cur_node.classification)
        return y_predict

    def evaluate(self, X_test, y_test, detailed_result=0):
        y_predict = self.predict(X_test)
        return self._calculate_metrics(y_predict, y_test, detailed_result)

    def _calculate_metrics(self, y_pred, y_true, detailed_result):
        """ If parameter detailed_result is False or 0, only prediction accuracy (Acc) will be returned.
            Otherwise, the returned result will be confusion matrix and prediction metrics list,
             in which only [Acc] for multiple classification and [Acc, Sn, Sp, Precision, MCC] for binary classification.
        """

        y_right = [1 for (y_p, y_t) in zip(y_pred, y_true) if y_p == y_t]
        acc = len(y_right) / len(y_pred)
        if not detailed_result:
            return acc

        con_matrix = confusion_matrix(y_pred, y_true)
        if len(self.y_kinds) > 2:
            return con_matrix, [acc]
        else:
            tn = con_matrix[0][0]
            fp = con_matrix[0][1]
            fn = con_matrix[1][0]
            tp = con_matrix[1][1]
            p = tp + fn
            n = tn + fp
            sn = tp / p if p > 0 else None
            sp = tn / n if n > 0 else None
            pre = (tp) / (tp + fp) if (tp + fp) > 0 else None
            mcc = 0
            tmp = sqrt(tp + fp) * sqrt(tp + fn) * sqrt(tn + fp) * sqrt(tn + fn)
            if tmp != 0:
                mcc = (tp * tn - fp * fn) / tmp
            return con_matrix, [acc, sn, sp, pre, mcc]

    def pruning(self, X_validation, y_validation, verbose=0):
        print('# Pruning tree, wait please...')
        print('Number of samples used for pruning:', len(X_validation))
        self._count_y_frequence_of_each_node(X_validation, y_validation)
        # store the currently unprocessed nodes
        node_queue = PriorityQueue()
        node_set = set()
        # get all leaf nodes
        leaf_node_list = self._get_all_leaf_nodes(self.root)
        # count tree node number
        all_node_num, leaf_node_num = self._count_node_number(self.root)
        print("There are {} tree nodes and {} leaf nodes before pruning.".format(all_node_num, leaf_node_num))

        # get the parents of all leaf nodes
        for leaf in leaf_node_list:
            parent = leaf.parent
            if parent != None and parent not in node_set:
                node_set.add(parent)
                node_queue.put(parent)

        # pruning
        best_acc = self.evaluate(X_validation, y_validation, detailed_result=0)
        while len(node_set) > 0:
            # get the deepest unprocessed node
            node = node_queue.get()
            node_queue.task_done()
            node_set.remove(node)

            if node.y_frequence_map != None:
                # set the current node to leaf node
                node.is_leaf = True

                # find the most frequent y value of the current node
                most_frequent_y = \
                    sorted([(x, node.y_frequence_map[x]) for x in node.y_frequence_map], key=lambda x: x[1])[-1][0]

                node.classification = most_frequent_y
                # performance validation
                val_acc = self.evaluate(X_validation, y_validation, detailed_result=0)
                if verbose:
                    print("Dealing with a TreeNode with depth being {}".format(node.depth))
                    print("Val Acc: ", val_acc)

                # update the best Acc
                if val_acc < best_acc:      # <= 0.94366 / <
                    # rollback, if val_acc doesn't improve
                    node.is_leaf = False
                    node.classification = None
                else:
                    best_acc = val_acc
                    if verbose:
                        print("This node has been translated into a leaf node.")
                        print("Current Best Acc: ", best_acc)

            # add the parent of current node to the unprocessed node queue
            if node.parent not in node_set:
                if node.parent != None:
                    node_set.add(node.parent)
                    node_queue.put(node.parent)
                elif node.parent is None and node.is_root == False:
                    print("We've found a non-root node has NO Parent!")

        # count tree node number
        all_node_num, leaf_node_num = self._count_node_number(self.root)
        print("There are {} tree nodes and {} leaf nodes after pruning.".format(all_node_num, leaf_node_num))

    def _get_all_leaf_nodes(self, root):
        leaf_node_list = []
        if root.is_leaf:
            leaf_node_list.append(root)
        else:
            for attr in root.descendants:
                leaf_node_list += self._get_all_leaf_nodes(root.descendants[attr])
        return leaf_node_list

    def _count_node_number(self, root):
        all_node_num = 1
        leaf_node_num = 0
        if root.is_leaf:
            leaf_node_num = 1
        else:
            for attr in root.descendants:
                child_node_num, child_leaf_num = self._count_node_number(root.descendants[attr])
                all_node_num += child_node_num
                leaf_node_num += child_leaf_num
        return all_node_num, leaf_node_num

    def _count_y_frequence_of_each_node(self, X, y):
        if not self.built:
            print("You should build the tree first by calling the 'fit' method with some train samples.")
            return None
        X_y = [list(xi) + [yi] for (xi, yi) in zip(X, y)]
        for x_y in X_y:
            cur_node = self.root
            while (not cur_node.is_leaf):
                # update the occurrence times of all kinds of y
                if cur_node.y_frequence_map is None:
                    cur_node.y_frequence_map = dict(zip(self.y_kinds, [0] * len(self.y_kinds)))
                cur_node.y_frequence_map[x_y[-1]] += 1

                x_attr_value = x_y[self.attr_position_map[cur_node.attr]]
                if (self.attr_is_discrete_map[cur_node.attr]):
                    cur_node = cur_node.descendants[x_attr_value]
                else:
                    if x_attr_value < cur_node.continuous_attr_value:
                        cur_node = cur_node.descendants['less']
                    else:
                        cur_node = cur_node.descendants['greater']  # performance test


def main():
    data_type_path = './datatypes.csv'  # a list to record whether an attribute is discrete or not
    train_data_path = './btrain.csv'
    test_data_path = './bvalidate.csv'
    random_seed = 0
    max_tree_depth = 35
    min_samples_split = 4
    max_continuous_attr_splits = 20
    validation_sample_num = 5000
    verbose = 0

    with open(data_type_path, 'r') as data_type:
        attr_type = data_type.read().strip().split(',')
        attr_type = [0 if x.lower() == 'false' else 1 for x in attr_type if len(x) > 0]

    def load_data(data_path, return_data_attrs=False):
        with open(data_path, 'r') as train_data:
            head = True
            samples = []
            completed_samples = []
            data_attrs = None
            for line in train_data:

                if head:
                    data_attrs = line.strip().split(',')
                    head = False
                else:
                    sample = line.strip().split(',')
                    samples.append(sample)
                    if '?' not in sample:
                        sample = [x if x.isalpha() else eval(x) for x in sample]
                        completed_samples.append(sample)

        if return_data_attrs:
            return completed_samples, data_attrs
        else:
            return completed_samples

    samples, data_attrs = load_data(train_data_path, return_data_attrs=True)
    test_samples = load_data(test_data_path)
    # shuffle samples
    np.random.seed(random_seed)
    np.random.shuffle(samples)

    validation_samples = samples[-validation_sample_num:]
    train_samples = samples[0:-validation_sample_num]

    X_train = [x[0:-1] for x in train_samples]
    y_train = [x[-1] for x in train_samples]
    X_validation = [x[0:-1] for x in validation_samples]
    y_validation = [x[-1] for x in validation_samples]
    X_test = [x[0:-1] for x in test_samples]
    y_test = [x[-1] for x in test_samples]

    tree = DTC45(max_depth=max_tree_depth, min_samples_split=min_samples_split,
                 max_continuous_attr_splits=max_continuous_attr_splits)
    tree.fit(X_train, y_train, attr_list=data_attrs[0:-1], attr_is_discrete=attr_type, verbose=verbose)

    print("Train Acc: {}".format(tree.evaluate(X_train, y_train, detailed_result=0)))
    con_matrix_before_pruning, performances_before_pruning = tree.evaluate(X_test, y_test, detailed_result=1)
    print("\nTest Acc before pruning: {}".format(performances_before_pruning[0]))
    print("Classification confusion_matrix:\n{}".format(con_matrix_before_pruning))
    print("Detailed performances [Acc, Sn, Sp, Pre, MCC]:\n{}\n".format(performances_before_pruning))

    tree.pruning(X_validation, y_validation, verbose=verbose)

    con_matrix_after_pruning, performances_after_pruning = tree.evaluate(X_test, y_test, detailed_result=1)
    print("\nTest Acc after pruning: {}".format(performances_after_pruning[0]))
    print("Classification confusion_matrix:\n{}".format(con_matrix_after_pruning))
    print("Detailed performances [Acc, Sn, Sp, Pre, MCC]:\n{}".format(performances_after_pruning))


# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    bank = pd.read_csv('../data/bank.csv')
    bank.to_csv(index=False)
    X = np.array(bank.ix[:,bank.columns[0:-1]], dtype=object)
    y = np.array(bank.ix[:,bank.columns[-1]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    attr_list = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_attris = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']


    dt = DTC45()
    dt.fit(X_train[:-5000,:], y_train[:-5000], attr_list, attr_is_discrete=[x in categorical_attris for x in attr_list])

    print(dt.evaluate(X_train, y_train))
    print(dt.evaluate(X_test, y_test))


    dt.pruning(X_train[-5000:,:], y_train[-5000:])
    print(dt.evaluate(X_train, y_train))
    print(dt.evaluate(X_test, y_test))