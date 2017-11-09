class Node():
    def __init__(self, idx):
        self.parent = None
        self.name = None
        self.idx = idx
        self.accepted_children = dict()
        self.classifier_id = None


with open('wordnet.is_a.txt') as f:
    raw_hierarchy_pairs = f.readlines()
    hierarchy_pairs = [raw_hierarchy_pair.strip().split(" ") for raw_hierarchy_pair in raw_hierarchy_pairs]

with open('words.txt') as f:
    raw_id_name_pairs = f.readlines()
    id_name_pairs = [raw_id_name_pair.strip().split("\t") for raw_id_name_pair in raw_id_name_pairs]
    id_name_dict = dict(id_name_pairs)

node_dict = dict()

def get_node(node_id):
    if node_id not in node_dict:
        node = Node(node_id)
        node.name = id_name_dict[node_id]
        node_dict[node_id] = node
    else:
        node = node_dict[node_id]
    return node

for hierarchy_pair in hierarchy_pairs:
    parent_id, child_id = hierarchy_pair
    child, parent = get_node(child_id), get_node(parent_id)
    child.parent = parent


with open('label_ids.txt') as f:
    raw_label_ids = f.readlines()
    label_ids = [raw_label_id.strip().split("\t") for raw_label_id in raw_label_ids]

root_nodes = dict()
for name, idx, classifier_id in label_ids:
    node = get_node(idx)
    node.classifier_id = classifier_id
    child_node = None
    while node is not None:
        if child_node is None:
            pass
        else:
            node.accepted_children[child_node.idx] = child_node
        child_node = node
        node = node.parent
    root_nodes[child_node.idx] = child_node


def travers_depth_first(node, depth):
    print (" " * depth * 4) + (node.name) +  ("" if node.classifier_id is None else ":" + node.classifier_id)
    for child_node in node.accepted_children.values():
        travers_depth_first(child_node, depth + 1)


for root_node in root_nodes.values():
    travers_depth_first(root_node, 0)