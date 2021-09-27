import json
import torch
import numpy as np

class HierarchyDistances():

    def __init__(self, hierarchy, distances, class_to_idx, max_levels=7, attack='NHA',
                 level=3):
        self.hierarchy = hierarchy
        self.distances = distances.distances
        self.class_to_idx = class_to_idx
        self.n_classes = len(self.class_to_idx.keys())
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        self.distance_matrix = np.zeros((len(class_to_idx.keys()), len(class_to_idx.keys())),
                                         dtype='uint8')

        self.hierarchy_levels = self.recursive_tree_class(hierarchy, 0, max_levels)
        self.hierarchy_levels = {7 - k: v for k, v in self.hierarchy_levels.items()}
        self.hierarchy_levels[0] = [[k] for k in range(self.n_classes)]

        for k, v in self.hierarchy_levels.items():
            self.hierarchy_levels[k] = []
            for classes in v:
                self.hierarchy_levels[k].append(torch.tensor(classes))
        
        for (k1, k2), v in self.distances.items():

            if (k1 not in class_to_idx.keys()) or (k2 not in class_to_idx.keys()):
                continue

            i1 = self.class_to_idx[k1]
            i2 = self.class_to_idx[k2]
            self.distance_matrix[i1, i2] = self.distance_matrix[i2, i1] = self.distances[(k1, k2) if k1 < k2 else (k2, k1)]

        self.distance_matrix = torch.from_numpy(self.distance_matrix)

        if attack == 'NHA':
            self.attack = self.get_logits_NHA_at_level
        elif attack == 'LHA':
            self.attack = self.get_logits_LHA_at_level
        elif attack == 'GHA':
            self.attack = self.get_logits_GHA_at_level

        self.level = level

        self.current_to_all = None
        self.current_n_classes = self.n_classes

    def get_logits(self, logits, target):
        return self.attack(logits, target)

    def recursive_tree_class(self, tree, level, max_levels):

        classes = []
        subtrees = {k: [] for k in range(level, max_levels + 1)}

        for subtree in tree:

            if isinstance(subtree, str):  # this is a class!
                classes.append(self.class_to_idx[subtree])
            else:  # this is a subtree
                subsubtrees = self.recursive_tree_class(subtree, level + 1, max_levels)

                for k, v in subsubtrees.items():
                    for subsubtree in v:
                        subsubtree = list(dict.fromkeys(subsubtree))  # remove duplicates
                        subtrees[k].append(subsubtree)
                        classes.extend(subsubtree)

        subtrees[level] = [list(dict.fromkeys(classes))]  # remove duplicates

        return subtrees

    def get_logits_NHA_at_level(self, logits, target):
        trees = self.hierarchy_levels[self.level]
        max_dim = len(trees)

        B = logits.size(0)

        new_logits = torch.zeros(B, max_dim, device=logits.device)
        new_target = torch.zeros_like(target)

        target = target.unsqueeze(dim=1)  # B x 1

        for idx, tree in enumerate(trees):
            eq = (tree.view(1, -1).expand(B, -1).to(target.device) == target)  # this tells us whether the label is within tree
            new_logits[:, idx] = logits[:, tree].max(dim=1)[0]

            # create a new target!
            new_target[eq.max(dim=1)[0]] = idx

        return new_logits, new_target

    def get_logits_LHA_at_level(self, logits, target):
        classes = (self.distance_matrix <= self.level)[target, :].to(target.device)
        B = logits.size(0)

        new_logits = torch.zeros_like(logits)
        new_logits.fill_(-float('inf'))  # -inf so when we compute the loss e⁻inf = 0

        # we fill with the logits values the new_logits array
        new_logits[classes] = logits[classes]

        return new_logits, target

    def get_logits_GHA_at_level(self, logits, target):
        classes = ((self.distance_matrix >= self.level) | torch.eye(self.distance_matrix.size(0), dtype=torch.bool))[target, :]
        B = logits.size(0)

        new_logits = torch.zeros_like(logits)
        new_logits.fill_(-float('inf'))  # -inf so when we compute the loss e⁻inf = 0

        # we fill with the logits values the new_logits array
        new_logits[classes] = logits[classes]

        return new_logits, target

    def get_transform_variables(self, target_level):

        class label_transform():
            def __init__(self, current_to_all):
                self.current_to_all = current_to_all

            def __call__(self, x):
                for current_class, leaf_classes in self.current_to_all.items():
                    if torch.any(leaf_classes == x):
                        return current_class

        trees = self.hierarchy_levels[target_level]
        self.current_n_classes = len(trees)
        none_current_to_all = self.current_to_all is None
        print(f'==> Setting new set of labels. Total classes: {self.current_n_classes}')

        # First time network initialization
        if none_current_to_all:
            self.current_to_all = {}
            for new_class, tree in enumerate(trees):
                self.current_to_all[new_class] = tree.view(1, -1).clone()
            return label_transform(self.current_to_all), None

        # else
        new_current_to_all = {}
        transition = {k: [] for k in self.current_to_all.keys()}
        for new_class, tree in enumerate(trees):

            # tree, tensor with shape n_classes2
            new_current_to_all[new_class] = tree.view(1, -1).clone()
            tree = tree.view(-1, 1)

            for current_class, leaf_classes in self.current_to_all.items():

                # this means that one of the classes
                # within current_class is on the tree
                if torch.any(tree == leaf_classes):
                    transition[current_class].append(new_class)

        self.current_to_all = new_current_to_all

        return label_transform(self.current_to_all), transition

    def initialize_new_classification_layer(self, model, target_level):

        lt, transition = self.get_transform_variables(target_level)
        
        if transition is None:
            in_features = model.model.fc[1].in_features
            model.model.fc[1] = torch.nn.Linear(in_features=in_features,
                                                out_features=self.current_n_classes,
                                                bias=True).cuda()
            return lt

        out_features = self.current_n_classes

        in_features = model.model.fc[1].in_features

        old_weight = model.model.fc[1].weight.data  # shape, out x in
        old_bias = model.model.fc[1].bias.data

        new_weight = torch.zeros(out_features, in_features).cuda()
        new_bias = torch.zeros(out_features).cuda()

        for current_class, new_class in transition.items():
            new_weight[new_class, :] = old_weight[current_class, :].detach()
            new_bias[new_class] = old_bias[current_class].detach()

        new_bias = new_bias.detach()
        new_weight = new_weight.detach()

        model.model.fc[1] = torch.nn.Linear(in_features, out_features, bias=True)

        model.model.fc[1].bias.data = new_bias
        model.model.fc[1].weight.data = new_weight
        model.model.fc[1].out_features = out_features

        return lt

    def get_curriculum(self, epochs):

        # n_classes = {k: len(v) for k, v in self.hierarchy_levels.items()}
        # n_classes = {k: n_classes[k] / self.n_classes for k in sorted(n_classes)[::-1]}
        # n_classes = {k: max(0.01, v) for k, v in n_classes.items()}

        schedule = [(6, 0.02),
                    (5, 0.04),
                    (4, 0.06),
                    (3, 0.15),
                    (2, 0.25),
                    (1, 0.35),
                    (0, 1.00)]
        curriculum = [(l, int(epochs * p)) for (l, p) in schedule]

        print('Curriculum created. The training will follow the following schedule:')
        print(curriculum)

        self.curriculum = curriculum

    def get_current_stage(self, epoch):

        if epoch == 0:
            return 6

        epochs = np.array([v[1] for v in self.curriculum])
        stages = np.array([v[0] for v in self.curriculum])

        pos = np.where(epochs < epoch)[0]
        if len(pos) == 0:
            return 6
        else:
            pos = pos[-1]
        return stages[pos + 1]
