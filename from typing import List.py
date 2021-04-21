from typing import List
from collections import defaultdict
import spacy
from spacy.pipeline import merge_entities

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(merge_entities)


from spacy import displacy
from spacy.tokens import Doc


def visualise_doc(doc: Doc):
    """
    Visualise both the dependency tree and entities in a spacy Doc.
    """
    displacy.render(doc, style="dep", options={"distance": 120}, jupyter=True)
    displacy.render(doc, style="ent", options={"distance": 120}, jupyter=True)


def visualise_subtrees(doc: Doc, subtrees: List[int]):

    words = [{"text": t.text, "tag": t.pos_} for t in doc]

    if not isinstance(subtrees[0], list):
        subtrees = [subtrees]

    for subtree in subtrees:
        arcs = []

        tree_indices = set(subtree)
        for index in subtree:

            token = doc[index]
            head = token.head
            if token.head.i == token.i or token.head.i not in tree_indices:
                continue

            else:
                if token.i < head.i:
                    arcs.append(
                        {
                            "start": token.i,
                            "end": head.i,
                            "label": token.dep_,
                            "dir": "left",
                        }
                    )
                else:
                    arcs.append(
                        {
                            "start": head.i,
                            "end": token.i,
                            "label": token.dep_,
                            "dir": "right",
                        }
                    )
        print("Subtree: ", subtree)
        displacy.render(
            {"words": words, "arcs": arcs},
            style="dep",
            options={"distance": 120},
            manual=True,
            jupyter=True
        )




def check_for_non_trees(dependency_triples: List[List[str]]):
    """
    A utility function which checks:

    1. The dependency triples you pass in are not self referential
    2. The triples you pass in form a single tree, with one root.
    3. There are no loops in the triples you pass in.

    # Parameters
    dependency_triples: List[List[str]]
        A list of [parent, relation, child] triples, which together
        form a tree that we would like to match on.

    # Returns
    root: str
        The root of the subtree
    parent_to_children: Dict[str, List[Tuple[str, str]]]
        A dictionary mapping parents to a list of their children,
        where the child is represented as a (relation, child) tuple.
    """

    parent_to_children = defaultdict(list)
    seen = set()
    has_incoming_edges = set()
    for (parent, rel, child) in dependency_triples:
        seen.add(parent)
        seen.add(child)
        has_incoming_edges.add(child)
        if parent == child:
            return None, None
        parent_to_children[parent].append((rel, child))

    # Only accept strictly connected trees with a single root.
    roots = seen.difference(has_incoming_edges)
    if len(roots) != 1:
        return None, None

    root = roots.pop()
    seen = {root}

    # Step 2: check that the tree doesn't have a loop:
    def contains_loop(node):
        has_loop = False
        for (_, child) in parent_to_children[node]:
            if child in seen:
                return True
            else:
                seen.add(child)
                has_loop = contains_loop(child)
            if has_loop:
                break

        return has_loop

    if contains_loop(root):
        return None, None

    return root, parent_to_children


def construct_pattern(dependency_triples: List[List[str]]):
    """
    Idea: add patterns to a matcher designed to find a subtree in a spacy dependency tree.
    Rules are strictly of the form "Parent --rel--> Child". To build this up, we add rules
    in DFS order, so that the parent nodes have already been added to the dict for each child
    we encounter.

    # Parameters
    dependency_triples: List[List[str]]
        A list of [parent, relation, child] triples, which together
        form a tree that we would like to match on.

    # Returns
    pattern:
        A json structure defining the match for the given tree, which
        can be passed to the spacy DependencyMatcher.

    """
    # Step 1: Build up a dictionary mapping parents to their children
    # in the dependency subtree. Whilst we do this, we check that there is
    # a single node which has only outgoing edges.

    root, parent_to_children = check_for_non_trees(dependency_triples)
    if root is None:
        return None

    def add_node(parent: str, pattern: List):

        for (rel, child) in parent_to_children[parent]:
            # First, we add the specification that we are looking for
            # an edge which connects the child to the parent.
            node = {
                "SPEC": {"NODE_NAME": child, "NBOR_RELOP": ">", "NBOR_NAME": parent}
            }
            # We want to match the relation exactly.
            token_pattern = {"DEP": rel}

            # Because we're working specifically with relation extraction in mind,
            # we'll use START_ENTITY and END_ENTITY as dummy placeholders in our
            # list of triples to indicate that we want to match a word which is contained
            # within an entity (or the entity itself if you have added the merge_entities pipe
            # to your pipeline before running the matcher).
            if child not in {"START_ENTITY", "END_ENTITY"}:
                token_pattern["ORTH"] = child
            else:
                token_pattern["ENT_TYPE"] = {"NOT_IN": [""]}

            node["PATTERN"] = token_pattern

            pattern.append(node)
            add_node(child, pattern)

    pattern = [{"SPEC": {"NODE_NAME": root}, "PATTERN": {"ORTH": root}}]
    add_node(root, pattern)

    return pattern


from spacy.matcher import DependencyMatcher


example = [["founded", "nsubj", "START_ENTITY"], ["founded", "dobj", "END_ENTITY"]]

pattern = construct_pattern(example)
matcher = DependencyMatcher(nlp.vocab)
matcher.add("pattern1", None, pattern)

doc1 = nlp("Bill Gates founded Microsoft.")
doc2 = nlp("Bill Gates, the Seattle Seahawks owner, founded Microsoft.")

match = matcher(doc1)[0]
subtree = match[1][0]
visualise_subtrees(doc1, subtree)

match = matcher(doc2)[0]
subtree = match[1][0]
visualise_subtrees(doc2, subtree)
